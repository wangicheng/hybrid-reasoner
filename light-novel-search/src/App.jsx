import { useState, useEffect, useRef } from 'react';
import Typewriter from 'typewriter-effect';
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
  const [engineData, setEngineData] = useState(null);
  const [terminalLines, setTerminalLines] = useState([]);
  const eventSourceRef = useRef(null);

  // Clean up streaming connection when component unmounts
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleSearch = (val) => {
    if (!val) return;

    // Clean up any stale connections before starting a new search
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setSearchState('fetching');
    setTerminalLines([]);
    setResults([]);
    setEngineData({ query: val });
    setPipelineStep(1);
    setTags([]);

    const url = `http://127.0.0.1:8000/api/search/stream?query=${encodeURIComponent(val)}&model_id=gemma-4-31b-it`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    setTerminalLines(['> 正在與 Nexus Engine 建立神經連結，準備送出查詢...']);
    
    // Simulate transitioning into the Semantic Parsing Layer after initial setup
    setTimeout(() => {
      setPipelineStep(2);
    }, 1000);

    eventSource.addEventListener('planner', (e) => {
      const data = JSON.parse(e.data);
      setSearchState('typing');
      setTerminalLines(prev => [
        ...prev,
        '> 系統連線成功，接收到引擎初步結果...',
        `> LLM 意圖解析：並行提取關鍵特徵中...`,
        `> <span style="color: #a855f7">發現標籤：${data.positive_terms.join(', ') || '無明顯標籤'}</span>`
      ]);
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
      setPipelineStep(3); // Move to Structure/Tag branch momentarily
      setTimeout(() => {
        setPipelineStep(prev => (prev === 3 ? 4 : prev)); // Then advance to Merge safely
      }, 1000);
    });

    eventSource.addEventListener('retrieval', (e) => {
      const data = JSON.parse(e.data);
      setTerminalLines(prev => [
        ...prev,
        `> 外部資料庫篩選：從向量與屬性資料庫召回 ${data.candidate_count} 筆候選作品...`
      ]);
      setEngineData(prev => ({
        ...prev,
        results: data.results || []
      }));
      setPipelineStep(prev => Math.max(prev, 5));
    });

    eventSource.addEventListener('post_filter', (e) => {
      const data = JSON.parse(e.data);
      setTerminalLines(prev => [
        ...prev,
        `> 規則過濾層：依據意圖硬性篩選完成，剩餘 ${data.filtered_count} 筆有效候選集...`
      ]);
      setEngineData(prev => ({
        ...prev,
        results: data.results || prev?.results || []
      }));
    });

    eventSource.addEventListener('rerank', (e) => {
      setTerminalLines(prev => [
        ...prev,
        `> LLM 精確排序：正在進行全方位深層交互評分中...`,
        `> <span style="color: #22c55e">重排序完成，Top 1: ${e.data.top_results?.[0]?.name || '計算中'}...</span>`
      ]);
      setEngineData(prev => ({
        ...prev,
        top_results: e.data.top_results
      }));
      setPipelineStep(prev => Math.max(prev, 6));
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

      setTerminalLines(prev => [...prev, '> 準備渲染全息卡片...']);
      setTimeout(() => {
        setResults(realResults);
        setSearchState('results');
        setPipelineStep(prev => Math.max(prev, 7));
      }, 800);
    });

    eventSource.onopen = () => {
      setTerminalLines(prev => {
        if (prev.some(line => line.includes("偵測到連線中斷"))) {
          return [...prev, '> <span style="color: #22c55e">✅ 連線已重新建立，繼續接收數據...</span>'];
        }
        return prev;
      });
    };

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      if (eventSource.readyState === EventSource.CONNECTING) {
        // Transient disconnect (e.g. browser throttled connection in background), auto-reconnecting
        setTerminalLines(prev => {
          if (prev.some(line => line.includes("偵測到連線中斷"))) {
            return prev;
          }
          return [...prev, '> <span style="color: #eab308">⚠️ 偵測到連線中斷，正在嘗試自動重新連線...</span>'];
        });
      } else {
        // Fatal disconnect (readyState === EventSource.CLOSED)
        setTerminalLines(prev => [...prev, '> <span style="color: #ef4444">❌ 連線錯誤：請檢查後端伺服器狀態</span>']);
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
        <div className={cn("text-center transition-all duration-700", searchState !== 'idle' ? "mb-4" : "mb-12")}>
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
        <div className={cn("w-full max-w-2xl transition-all duration-700", searchState !== 'idle' ? "-translate-y-8" : "")}>
          {searchState === 'idle' && (
            <VanishInput
              placeholders={["找一本沒有悲劇結尾的快節奏異世界小說...", "主角從第一章就開掛無敵的龍傲天...", "有傲嬌青梅竹馬的戀愛喜劇..."]}
              onSubmit={handleSearch}
            />
          )}
        </div>

        {/* Loading State */}
        {searchState === 'fetching' && (
          <div className="mt-12 text-cyan-400 font-mono animate-pulse">
            &gt; 正在與 Nexus Engine 建立神經連結，準備送出查詢...
          </div>
        )}

        {/* Terminal/Typewriter Status */}
        <div className={cn(
          "w-full max-w-3xl glass-strong rounded-lg font-mono text-cyan-400 transition-all duration-1000",
          (searchState === 'idle' || searchState === 'fetching') ? "opacity-0 translate-y-4 h-0 overflow-hidden p-0 mt-0" :
            searchState === 'results' ? "opacity-60 text-sm mt-2 p-4" : "opacity-100 translate-y-0 mt-8 p-6 text-base"
        )}>
          {searchState !== 'idle' && searchState !== 'fetching' && (
            <div className="flex flex-col gap-2">
              {terminalLines.map((line, idx) => (
                <div 
                  key={idx} 
                  className="animate-in fade-in slide-in-from-left-2 duration-300"
                  dangerouslySetInnerHTML={{ __html: line }}
                />
              ))}
              <div className="w-2 h-5 bg-cyan-400 animate-pulse inline-block" />
            </div>
          )}
        </div>

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
