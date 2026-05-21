import { useState, useEffect, useRef } from 'react';
import Tilt from 'react-parallax-tilt';
import { BackgroundBeams } from './components/BackgroundBeams';
import { SparklesCore } from './components/SparklesCore';
import { VanishInput } from './components/VanishInput';
import { EnginePipeline } from './components/EnginePipeline';
import { cn } from './lib/utils';
import './index.css';

// Mock data for search results
const mockResults = [
  {
    id: 1,
    title: "刀劍神域",
    cover: "https://images.unsplash.com/photo-1542831371-29b0f74f9713?q=80&w=400&auto=format&fit=crop",
    score: 98,
    tags: ["虛擬實境", "戰鬥", "戀愛"]
  },
  {
    id: 2,
    title: "Re:從零開始的異世界生活",
    cover: "https://images.unsplash.com/photo-1580130281320-0ef0754f2bf7?q=80&w=400&auto=format&fit=crop",
    score: 92,
    tags: ["異世界", "心理戰", "奇幻"]
  },
  {
    id: 3,
    title: "Overlord 不死者之王",
    cover: "https://images.unsplash.com/photo-1605806616949-1e87b487cb2a?q=80&w=400&auto=format&fit=crop",
    score: 85,
    tags: ["黑暗奇幻", "魔法", "反英雄"]
  }
];

function App() {
  const [searchState, setSearchState] = useState('idle'); // 'idle', 'fetching', 'typing', 'results'
  const [tags, setTags] = useState([]);
  const [results, setResults] = useState([]);
  const [pipelineStep, setPipelineStep] = useState(0);
  const [targetStep, setTargetStep] = useState(0);
  const [pendingResults, setPendingResults] = useState([]);
  const [engineData, setEngineData] = useState(null);
  const eventSourceRef = useRef(null);
  const targetStepRef = useRef(0);
  const pendingResultsRef = useRef([]);
  const intervalActiveRef = useRef(false);

  // Helper: update targetStep state AND ref synchronously to avoid stale reads
  const advanceTargetStep = (step) => {
    targetStepRef.current = step;
    setTargetStep(step);
  };

  // Helper: update pendingResults state AND ref synchronously
  const updatePendingResults = (results) => {
    pendingResultsRef.current = results;
    setPendingResults(results);
  };

  // Clean up streaming connection when component unmounts
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Progressive smooth transition logic for pipeline steps
  // Uses a ref flag instead of searchState dependency to avoid interval restarts
  useEffect(() => {
    const interval = setInterval(() => {
      if (!intervalActiveRef.current) return;

      setPipelineStep(prev => {
        const target = targetStepRef.current;
        if (prev < target) {
          const next = prev + 1;
          if (next === 4 && target >= 5) {
            // Smoothly and quickly transition to step 5 (150ms visual flash of the merge node)
            setTimeout(() => {
              setPipelineStep(p => p === 4 ? 5 : p);
            }, 150);
          }
          if (next === 7) {
            setResults(pendingResultsRef.current);
            setSearchState('results');
          }
          return next;
        }
        return prev;
      });
    }, 600);

    return () => clearInterval(interval);
  }, []); // Empty deps: interval runs forever, activation controlled by ref

  const handleSearch = (val) => {
    if (!val) return;

    // Clean up any stale connections before starting a new search
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setSearchState('fetching');
    setResults([]);
    updatePendingResults([]);
    setEngineData({ query: val });
    setPipelineStep(1);
    advanceTargetStep(2); // Start parsing phase transition immediately
    setTags([]);
    intervalActiveRef.current = true;

    const url = `http://127.0.0.1:8000/api/search/stream?query=${encodeURIComponent(val)}&model_id=gemma-4-31b-it`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener('semantic_understanding', (e) => {
      const data = JSON.parse(e.data);
      console.log(">>> [SSE Event] semantic_understanding:", data);
      setEngineData(prev => ({
        ...prev,
        search_terms: data.semantic_query_text,
        intent_summary: data.intent_summary,
        parsed_criteria: [
          ...(data.positive_concepts || []).map(concept => ({
            name: "semantic_similarity",
            is_negative: false,
            parameters: { query_text: concept }
          })),
          ...(data.negative_concepts || []).map(concept => ({
            name: "semantic_similarity",
            is_negative: true,
            parameters: { query_text: concept }
          }))
        ]
      }));
      advanceTargetStep(3); // Progress to step 3 (Structure & Tag branch)
    });

    eventSource.addEventListener('planner', (e) => {
      const data = JSON.parse(e.data);
      console.log(">>> [SSE Event] planner:", data);
      setSearchState('typing');
      setTags(data.positive_terms);
      setEngineData(prev => ({
        ...prev,
        ...data,
        tag_intent: {
          positive_terms: data.positive_terms,
          negative_terms: data.negative_terms,
          fuzzy_positive_terms: data.fuzzy_positive_terms || [],
          fuzzy_negative_terms: data.fuzzy_negative_terms || []
        }
      }));
      advanceTargetStep(5); // Progress directly to step 5 (Retrieval) as Requirement Merging completes instantly
    });

    eventSource.addEventListener('retrieval', (e) => {
      const data = JSON.parse(e.data);
      console.log(">>> [SSE Event] retrieval:", data);
      setEngineData(prev => ({
        ...prev,
        candidate_count: data.candidate_count,
        recall_tags: data.recall_tags
      }));
      advanceTargetStep(5); // Progress to step 5 (Retrieval)
    });

    eventSource.addEventListener('post_filter', (e) => {
      const data = JSON.parse(e.data);
      console.log(">>> [SSE Event] post_filter:", data);
      
      const preRerankCandidates = (data.pre_rerank_candidates || []).map((r) => {
        const coverUrl = (() => {
          const rawCover = r.cover;
          if (!rawCover) {
            return "https://images.unsplash.com/photo-1542831371-29b0f74f9713?q=80&w=400&auto=format&fit=crop";
          }
          if (rawCover.startsWith("http")) {
            return rawCover;
          }
          return `https://czbooks.net${rawCover}`;
        })();
        return {
          id: r.id,
          title: r.name,
          cover: coverUrl,
        };
      });

      setEngineData(prev => ({
        ...prev,
        filtered_count: data.filtered_count,
        pre_rerank_candidates: preRerankCandidates
      }));
      advanceTargetStep(6); // Enter reranker phase (LLM Rerank)
    });

    eventSource.addEventListener('rerank', (e) => {
      const data = JSON.parse(e.data);
      console.log(">>> [SSE Event] rerank:", data);
      setEngineData(prev => ({
        ...prev,
        top_results: data.top_results
      }));
      advanceTargetStep(6); // Progress to step 6 (Rerank)
    });

    eventSource.addEventListener('complete', (e) => {
      const data = JSON.parse(e.data);
      console.log(">>> [SSE Event] complete:", data);
      // Close EventSource immediately to avoid connection reset/drop errors from server teardown
      eventSource.close();
      if (eventSourceRef.current === eventSource) {
        eventSourceRef.current = null;
      }
      
      setEngineData(prev => ({
        ...prev,
        ...data
      }));
      
      const realResults = (data.results || []).map((r) => {
        const coverUrl = (() => {
          const rawCover = r.item?.cover_url || r.item?.cover;
          if (!rawCover) {
            return "https://images.unsplash.com/photo-1542831371-29b0f74f9713?q=80&w=400&auto=format&fit=crop";
          }
          if (rawCover.startsWith("http")) {
            return rawCover;
          }
          return `https://czbooks.net${rawCover}`;
        })();
        return {
          id: r.item?.id || Math.random(),
          title: r.item?.name || "未知書名",
          cover: coverUrl,
          score: Math.round(r.score * 100),
          vector_score: r.vector_score,
          bm25_score: r.bm25_score,
          tag_vector_score: r.tag_vector_score,
          tags: r.item?.tags || [],
          author: r.item?.author || "未知作者",
          status: r.item?.publish_status || r.item?.status || "未知狀態",
          words_total: r.item?.words_total || 0,
          intro: r.item?.intro || "暫無小說簡介。",
          explanation: r.explanation || ""
        };
      });

      updatePendingResults(realResults);
      advanceTargetStep(7); // Trigger final step 7 (Holographic Results presentation)
    });

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      // 伺服器斷線或連線錯誤時，直接關閉 EventSource，不進行自動重連
      eventSource.close();
      if (eventSourceRef.current === eventSource) {
        eventSourceRef.current = null;
      }
      intervalActiveRef.current = false;
      setSearchState('idle');
    };
  };

  return (
    <div className="relative min-h-screen bg-void text-zinc-100 font-display selection:bg-purple-500/30">
      {/* Background Layer */}
      <div className="fixed inset-0 z-0">
        <BackgroundBeams className="opacity-50" />
        <div className="absolute inset-0 bg-void/80 backdrop-blur-[2px]"></div>
        <SparklesCore
          id="tsparticlesfullpage"
          background="transparent"
          minSize={0.6}
          maxSize={1.4}
          particleDensity={50}
          className="w-full h-full absolute inset-0 opacity-30"
          particleColor="#a855f7"
        />
      </div>

      {/* Main Content */}
      <div className={cn("relative z-10 flex flex-col items-center min-h-screen px-4 pb-24 transition-all duration-700", searchState !== 'idle' ? "pt-8" : "pt-32")}>

        {/* Header */}
        <div className={cn("text-center transition-all duration-700", searchState !== 'idle' ? "mb-8" : "mb-12")}>
          <h1 className={cn("font-bold tracking-tight gradient-text pb-2 transition-all duration-700", 
            searchState !== 'idle' ? "text-3xl md:text-4xl mb-1" : "text-5xl md:text-7xl mb-4"
          )}>
            基於 RAG 與混合檢索的輕小說查詢系統
          </h1>
          {searchState === 'idle' && (
            <p className="text-zinc-400 text-lg md:text-xl max-w-2xl mx-auto animate-in fade-in duration-500">
              結合 LLM 與規則導向的輕小說語意搜尋引擎
            </p>
          )}
        </div>

        {/* Search Section */}
        <div className={cn("w-full max-w-2xl transition-all duration-700", searchState !== 'idle' ? "mt-2" : "")}>
          <VanishInput
            placeholders={["找一本沒有悲劇結尾的快節奏異世界小說...", "主角從第一章就開掛無敵的龍傲天...", "有傲嬌青梅竹馬的戀愛喜劇..."]}
            onSubmit={handleSearch}
          />
        </div>

        {/* Loading State */}
        {searchState === 'fetching' && (
          <div className="mt-12 text-cyan-400 font-mono animate-pulse">
            &gt; 正在與 Nexus Engine 建立神經連結，準備送出查詢...
          </div>
        )}



        {/* Pipeline Visualization */}
        {searchState !== 'idle' && (
          <div className="w-full max-w-5xl mt-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <EnginePipeline currentStep={pipelineStep} engineData={engineData} results={results} />
          </div>
        )}


      </div>
    </div>
  );
}

export default App;
