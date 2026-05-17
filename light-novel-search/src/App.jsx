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

  // Clean up streaming connection when component unmounts
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Progressive smooth transition logic for pipeline steps
  useEffect(() => {
    if (pipelineStep < targetStep) {
      const timer = setTimeout(() => {
        setPipelineStep(prev => {
          const next = prev + 1;
          if (next === 7 && pendingResults.length > 0) {
            setResults(pendingResults);
            setSearchState('results');
          }
          return next;
        });
      }, 600);
      return () => clearTimeout(timer);
    }
  }, [pipelineStep, targetStep, pendingResults]);

  const handleSearch = (val) => {
    if (!val) return;

    // Clean up any stale connections before starting a new search
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setSearchState('fetching');
    setResults([]);
    setPendingResults([]);
    setEngineData({ query: val });
    setPipelineStep(1);
    setTargetStep(2); // Start parsing phase transition immediately
    setTags([]);

    const url = `http://127.0.0.1:8000/api/search/stream?query=${encodeURIComponent(val)}&model_id=gemma-4-31b-it`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener('planner', (e) => {
      const data = JSON.parse(e.data);
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
      setTargetStep(4); // Trigger progressive steps: 2 -> 3 (Structure/Tag) -> 4 (Merge)
    });

    eventSource.addEventListener('retrieval', (e) => {
      const data = JSON.parse(e.data);
      setEngineData(prev => ({
        ...prev,
        results: data.results || []
      }));
      setTargetStep(5); // Progress to step 5 (Retrieval)
    });

    eventSource.addEventListener('post_filter', (e) => {
      const data = JSON.parse(e.data);
      setEngineData(prev => ({
        ...prev,
        results: data.results || prev?.results || []
      }));
    });

    eventSource.addEventListener('rerank', (e) => {
      setEngineData(prev => ({
        ...prev,
        top_results: e.data.top_results
      }));
      setTargetStep(6); // Progress to step 6 (Rerank)
    });

    eventSource.addEventListener('complete', (e) => {
      // Close EventSource immediately to avoid connection reset/drop errors from server teardown
      eventSource.close();
      if (eventSourceRef.current === eventSource) {
        eventSourceRef.current = null;
      }
      
      const data = JSON.parse(e.data);
      setEngineData(data);
      
      const realResults = (data.results || []).map((r) => {
        const coverUrl = r.item?.cover ? `https://czbooks.net${r.item.cover}` : "https://images.unsplash.com/photo-1542831371-29b0f74f9713?q=80&w=400&auto=format&fit=crop";
        return {
          id: r.item?.id || Math.random(),
          title: r.item?.name || "未知書名",
          cover: coverUrl,
          score: Math.round(r.score * 100),
          tags: (r.item?.tags || []).slice(0, 4)
        };
      });

      setPendingResults(realResults);
      setTargetStep(7); // Trigger final step 7 (Holographic Results presentation)
    });

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      if (eventSource.readyState !== EventSource.CONNECTING) {
        // Fatal disconnect (readyState === EventSource.CLOSED)
        eventSource.close();
        if (eventSourceRef.current === eventSource) {
          eventSourceRef.current = null;
        }
      }
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
