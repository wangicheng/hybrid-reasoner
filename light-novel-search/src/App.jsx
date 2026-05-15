import { useState, useEffect } from 'react';
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

  const handleSearch = (val) => {
    if (!val) return;
    setSearchState('fetching');
    setTerminalLines([]);
    setResults([]);
    setEngineData(null);
    setPipelineStep(1);
    setTags([]);

    const url = `http://127.0.0.1:8000/api/search/stream?query=${encodeURIComponent(val)}&model_id=gemma-4-31b-it`;
    const eventSource = new EventSource(url);

    setTerminalLines(['> 正在與 Nexus Engine 建立神經連結，準備送出查詢...']);
    
    // Simulate transitioning into the Parallel Parsing Layer after initial setup
    setTimeout(() => {
      setPipelineStep(2);
    }, 1500);

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
      setPipelineStep(3); // Planner completes steps 2 and 3
    });

    eventSource.addEventListener('retrieval', (e) => {
      const data = JSON.parse(e.data);
      setTerminalLines(prev => [
        ...prev,
        `> 外部資料庫篩選：從向量與屬性資料庫召回 ${data.candidate_count} 筆候選作品...`
      ]);
      setPipelineStep(4);
    });

    eventSource.addEventListener('post_filter', (e) => {
      const data = JSON.parse(e.data);
      setTerminalLines(prev => [
        ...prev,
        `> 規則過濾層：依據意圖硬性篩選完成，剩餘 ${data.filtered_count} 筆有效候選集...`
      ]);
    });

    eventSource.addEventListener('rerank', (e) => {
      setTerminalLines(prev => [
        ...prev,
        `> LLM 精確排序：正在進行全方位深層交互評分中...`,
        `> <span style="color: #22c55e">重排序完成，Top 1: ${e.data.top_results?.[0]?.name || '計算中'}...</span>`
      ]);
      setPipelineStep(5);
    });

    eventSource.addEventListener('complete', (e) => {
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
        setPipelineStep(5);
        eventSource.close();
      }, 800);
    });

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      setTerminalLines(prev => [...prev, '> <span style="color: #ef4444">連線錯誤：請檢查後端伺服器狀態</span>']);
      eventSource.close();
    };
  };

  return (
    <div className="relative min-h-screen bg-void text-zinc-100 font-display overflow-hidden selection:bg-purple-500/30">
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
      <div className="relative z-10 flex flex-col items-center min-h-screen pt-32 px-4">

        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4 gradient-text pb-2">
            Nexus 搜尋引擎
          </h1>
          <p className="text-zinc-400 text-lg md:text-xl max-w-2xl mx-auto">
            結合 LLM 與規則導向的輕小說語意搜尋引擎
          </p>
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
            <EnginePipeline currentStep={pipelineStep} engineData={engineData} />
          </div>
        )}

        {/* Neon Tags */}
        {tags.length > 0 && (
          <div className="flex gap-4 mt-8 flex-wrap justify-center animate-in fade-in slide-in-from-bottom-4 duration-700">
            {tags.map((tag, idx) => (
              <span
                key={idx}
                className="px-5 py-2 text-green-400 border border-green-500 shadow-[0_0_15px_rgba(34,197,94,0.3)] bg-green-500/10 rounded-full font-medium tracking-wide text-sm"
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Results Grid */}
        {searchState === 'results' && (
          <div className="w-full max-w-6xl mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-32">
            {results.map((book, idx) => (
              <div
                key={book.id}
                className="animate-in fade-in zoom-in-95 duration-700 fill-mode-both"
                style={{ animationDelay: `${idx * 150}ms` }}
              >
                <Tilt
                  tiltMaxAngleX={10}
                  tiltMaxAngleY={10}
                  perspective={1000}
                  scale={1.02}
                  transitionSpeed={2000}
                  className="h-full"
                >
                  <div className="glass h-full rounded-2xl p-4 flex flex-col relative group overflow-hidden border border-white/5 hover:border-purple-500/50 transition-colors">
                    {/* Cover Image Placeholder */}
                    <div className="w-full h-64 bg-zinc-800 rounded-xl mb-4 overflow-hidden relative">
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent z-10"></div>
                      <img src={book.cover} alt={book.title} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity duration-500 group-hover:scale-105" />

                      {/* Score Badge */}
                      <div className="absolute bottom-3 right-3 z-20 flex items-center justify-center w-14 h-14 rounded-full bg-black/50 backdrop-blur-md border border-cyan-500/50 glow-cyan">
                        <span className="text-cyan-400 font-bold">{book.score}%</span>
                        {/* Circular Progress SVG could go here */}
                        <svg className="absolute inset-0 w-full h-full -rotate-90">
                          <circle cx="28" cy="28" r="26" fill="none" stroke="rgba(0,240,255,0.2)" strokeWidth="2" />
                          <circle cx="28" cy="28" r="26" fill="none" stroke="#00f0ff" strokeWidth="2" strokeDasharray="163" strokeDashoffset={163 - (163 * book.score) / 100} className="transition-all duration-1000 delay-500" />
                        </svg>
                      </div>
                    </div>

                    {/* Info */}
                    <div className="flex-grow z-20">
                      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors">{book.title}</h3>
                      <div className="flex flex-wrap gap-2">
                        {book.tags.map(t => (
                          <span key={t} className="text-xs px-2 py-1 rounded bg-white/10 text-zinc-300">
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Hover Glow Effect */}
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
                  </div>
                </Tilt>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
