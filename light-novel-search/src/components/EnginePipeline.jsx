import { useState, useEffect } from "react";
import { cn } from "../lib/utils";
import { motion, AnimatePresence } from "framer-motion";

export function EnginePipeline({ currentStep, engineData }) {
  const [selectedStep, setSelectedStep] = useState(1);

  // Auto-select the current active step as it progresses
  useEffect(() => {
    if (currentStep > 0 && currentStep <= 5) {
      setSelectedStep(currentStep);
    }
  }, [currentStep]);

  const steps = [
    { id: 1, title: "輸入階段", desc: "查詢 (Query)" },
    { id: 2, title: "並行解析層", desc: "Parallel Parsing" },
    { id: 3, title: "需求合併", desc: "Requirement Merging" },
    { id: 4, title: "檢索與過濾", desc: "外部資料庫篩選" },
    { id: 5, title: "後處理與輸出", desc: "精確排序與輸出" },
  ];

  const getDetails = (stepId) => {
    // ... [existing getDetails logic retained below, we will insert it directly]
    if (!engineData) return { input: "等待引擎回應中...", output: "處理中..." };

    switch (stepId) {
      case 1:
        return {
          input: `> 系統就緒：\n準備接收使用者輸入並開始處理。\n> 使用者輸入：\n「${engineData.query}」`,
          output: `> 階段完成：\n已進入查詢管道，準備進行並行解析。\n原始輸入字串已快取。`
        };
      case 2:
        return {
          input: `> 語意理解：提取核心意圖\n> 結構過濾：解析特定欄位、分類\n> 標籤投影：映射至小說標籤庫`,
          output: JSON.stringify({
            search_terms: engineData.search_terms,
            generated_keywords: engineData.generated_keywords,
            tags: engineData.tag_intent?.positive_terms || [],
            negative_tags: engineData.tag_intent?.negative_terms || []
          }, null, 2)
        };
      case 3:
        return {
          input: `> 準備彙整下列分支結果：\n1. 語意脈絡\n2. 結構化限制\n3. 目標標籤集合`,
          output: `> 統一檢索需求：\n向量查詢參數與屬性過濾條件已生成，準備交予資料庫進行雙路徑檢索。`
        };
      case 4:
        return {
          input: `> 向量查詢 (Vector Query)：\n[ 語意嵌入轉換: "${engineData.search_terms}" ]\n> 屬性查詢 (Attribute Query)：\n強制標籤過濾與匹配，包含: [${(engineData.tag_intent?.positive_terms || []).join(", ")}]`,
          output: `> 執行結果：\n已從底層向量資料庫召回 ${engineData.results?.length || 0} 筆候選名單，準備進行預先過濾。`
        };
      case 5:
        return {
          input: `> 重排序模型 (Reranker)：\n將使用者完整 Query 與候選小說進行深層注意力機制交互運算...`,
          output: `> 最終排序完成：\n` + 
            (engineData.results?.slice(0, 3).map((r, i) => `${i+1}. ${r.item?.name || "未知"} (評分: ${Math.round(r.score*100)}%)`).join("\n") || "")
        };
      default:
        return { input: "", output: "" };
    }
  };

  // Node geometries and X,Y coordinates
  const W_RECT = 130, H_RECT = 48;
  const W_DIV = 20, H_DIV = 80;
  const D_DIAM = 64; 
  const W_OVAL = 110, H_OVAL = 40;
  const W_RES = 100, H_RES = 48;

  const X_IN = 80;
  const X_DIV = 200;
  const X_BR = 380; 
  const X_MRG = 560;
  const X_RUL = 700;
  const X_SCO = 860;
  const X_LLM = 1020;
  const X_RES = 1160;

  const Y_T = 100, Y_M = 220, Y_B = 340, Y_O = 60;

  // Path commands (M start L end)
  const lines = [
    { id: 'l1', d: `M 145 ${Y_M} L 185 ${Y_M}`, activeAt: 2 }, 
    { id: 'l2a', d: `M 210 ${Y_M} C 260 220, 260 100, 310 100`, activeAt: 2 }, 
    { id: 'l2b', d: `M 210 ${Y_M} L 310 ${Y_M}`, activeAt: 2 }, 
    { id: 'l2c', d: `M 210 ${Y_M} C 260 220, 260 340, 310 340`, activeAt: 2 },
    { id: 'l3a', d: `M 445 ${Y_T} C 500 100, 500 220, 523 220`, activeAt: 3 }, 
    { id: 'l3b', d: `M 445 ${Y_M} L 523 ${Y_M}`, activeAt: 3 }, 
    { id: 'l3c', d: `M 445 ${Y_B} C 500 340, 500 220, 523 220`, activeAt: 3 }, 
    { id: 'l4', d: `M 592 ${Y_M} L 630 ${Y_M}`, activeAt: 4 }, 
    { id: 'l4a', d: `M ${X_RUL} 80 L ${X_RUL} 191`, activeAt: 4 }, 
    { id: 'l5', d: `M 765 ${Y_M} L 790 ${Y_M}`, activeAt: 4 }, 
    { id: 'l5a', d: `M ${X_SCO} 80 L ${X_SCO} 191`, activeAt: 4 }, 
    { id: 'l6', d: `M 925 ${Y_M} L 950 ${Y_M}`, activeAt: 5 }, 
    { id: 'l7', d: `M 1085 ${Y_M} L 1105 ${Y_M}`, activeAt: 5 }, 
  ];

  const nodeStyle = (stepId, isActiveAnim = false) => {
    const isActive = currentStep === stepId;
    const isPast = currentStep > stepId;
    const isPending = currentStep < stepId;
    const isSelected = selectedStep === stepId;

    let border = "border-zinc-700 text-zinc-400";
    let bg = "bg-zinc-900/80";
    if (isActive) {
      border = "border-cyan-400 text-cyan-300 shadow-[0_0_15px_rgba(0,240,255,0.3)]";
      bg = "bg-cyan-900/20";
    } else if (isPast) {
      border = "border-purple-500/80 text-purple-300";
      bg = "bg-purple-900/10";
    }
    
    if (isSelected) {
      border += " ring-2 ring-white/30 ring-offset-2 ring-offset-void";
      bg += " !bg-zinc-800";
    }

    return cn(
      "absolute flex items-center justify-center border-2 backdrop-blur-md transition-all duration-500 cursor-pointer text-[13px] font-bold z-20 hover:border-white/50",
      border, bg,
      isPending ? "opacity-50" : "opacity-100"
    );
  };

  return (
    <div className="w-full py-8 mt-4 hidden md:block">
      <div className="w-full overflow-x-auto pb-8 custom-scrollbar">
        <div className="relative w-[1240px] h-[400px] mx-auto shrink-0 select-none">
          
          {/* SVG Connecting Lines */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-10" strokeWidth="2" fill="none">
            <defs>
              <marker id="arrow-pending" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="fill-zinc-700" />
              </marker>
              <marker id="arrow-active" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="fill-cyan-400" />
              </marker>
              <marker id="arrow-past" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="fill-purple-500" />
              </marker>
            </defs>
            
            {lines.map((line) => {
              const status = currentStep === line.activeAt ? 'active' : currentStep > line.activeAt ? 'past' : 'pending';
              const color = status === 'active' ? 'stroke-cyan-400' : status === 'past' ? 'stroke-purple-500/80' : 'stroke-zinc-700';
              const marker = `url(#arrow-${status})`;
              return (
                <path
                  key={line.id}
                  d={line.d}
                  className={cn("transition-colors duration-500 delay-150", color)}
                  markerEnd={marker}
                />
              );
            })}
          </svg>

          {/* Background Container (Parallel Parsing Layer) */}
          <div 
            className="absolute border border-dashed border-zinc-700/50 rounded-xl bg-zinc-900/20 z-0 pointer-events-auto cursor-pointer hover:border-zinc-500/50 transition-colors"
            style={{ left: 150, top: 50, width: 335, height: 340 }}
            onClick={() => setSelectedStep(2)}
          >
            <div className="absolute -top-3 left-4 px-2 bg-void text-xs text-zinc-500 font-mono tracking-widest">
              PARALLEL PARSING LAYER
            </div>
            {selectedStep === 2 && <div className="absolute -left-1 -bottom-2 w-2 h-2 rounded-full bg-white animate-pulse" />}
          </div>

          {/* Small Labels on paths */}
          <div className="absolute z-10 text-[10px] text-zinc-400 bg-void border border-zinc-800 px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap" style={{ left: 260, top: 120, transform: 'translate(-50%, -50%)' }}>
            經由語意脈絡引導
          </div>
          <div className="absolute z-10 text-[10px] text-zinc-400 bg-void border border-zinc-800 px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap" style={{ left: 260, top: 310, transform: 'translate(-50%, -50%)' }}>
            經由語意脈絡引導
          </div>

          {/* Nodes */}
          {/* 1. Input */}
          <div 
            className={cn(nodeStyle(1), "rounded-md")}
            style={{ left: X_IN, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(1)}
          >
            查詢 (Query)
          </div>

          {/* 2. Parallel Layer Nodes */}
          <div 
            className={cn(nodeStyle(2), "rounded-sm bg-zinc-800")}
            style={{ left: X_DIV, top: Y_M, width: W_DIV, height: H_DIV, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(2)}
          />
          <div 
            className={cn(nodeStyle(2), "rounded-md")}
            style={{ left: X_BR, top: Y_T, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(2)}
          >
            結構過濾分支
          </div>
          <div 
            className={cn(nodeStyle(2), "rounded-md")}
            style={{ left: X_BR, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(2)}
          >
            語意理解分支
          </div>
          <div 
            className={cn(nodeStyle(2), "rounded-md")}
            style={{ left: X_BR, top: Y_B, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(2)}
          >
            標籤投影分支
          </div>

          {/* 3. Merge Diamond */}
          <div 
            className={cn(nodeStyle(3), "rotate-45")}
            style={{ left: X_MRG, top: Y_M, width: D_DIAM, height: D_DIAM, transform: 'translate(-50%, -50%) rotate(45deg)' }}
            onClick={() => setSelectedStep(3)}
          >
            <div className="-rotate-45 text-[11px] text-center leading-tight">需求<br/>合併</div>
          </div>

          {/* 4. Filter & Retrieval Layer */}
          <div 
            className={cn(nodeStyle(4), "rounded-full shadow-inner")}
            style={{ left: X_RUL, top: Y_O, width: W_OVAL, height: H_OVAL, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(4)}
          >
            屬性資料庫
          </div>
          <div 
            className={cn(nodeStyle(4), "rounded-md")}
            style={{ left: X_RUL, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(4)}
          >
            規則過濾
          </div>

          <div 
            className={cn(nodeStyle(4), "rounded-full shadow-inner")}
            style={{ left: X_SCO, top: Y_O, width: W_OVAL, height: H_OVAL, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(4)}
          >
            向量資料庫
          </div>
          <div 
            className={cn(nodeStyle(4), "rounded-md")}
            style={{ left: X_SCO, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(4)}
          >
            評分與融合
          </div>

          {/* 5. Post Processing */}
          <div 
            className={cn(nodeStyle(5), "rounded-md")}
            style={{ left: X_LLM, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(5)}
          >
            LLM 重排
          </div>
          <div 
            className={cn(nodeStyle(5), "rounded border-zinc-600 bg-zinc-800")}
            style={{ left: X_RES, top: Y_M, width: W_RES, height: H_RES, transform: 'translate(-50%, -50%)' }}
            onClick={() => setSelectedStep(5)}
          >
            結果
          </div>
        </div>
      </div>

      {/* Details Panel */}
      <div className="mt-8 max-w-5xl mx-auto h-48 relative">
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedStep}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 w-full"
          >
            {steps.map(s => {
              if (s.id !== selectedStep) return null;
              const details = getDetails(s.id);
              return (
                <div key={s.id} className="grid grid-cols-2 gap-6 h-full">
                  <div className="glass-strong rounded-lg p-5 border-l-4 border-l-cyan-500 flex flex-col bg-zinc-900/60 shadow-lg">
                    <h5 className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">{s.title} - Input / Task</h5>
                    <pre className="font-mono text-[11px] text-zinc-300 whitespace-pre-wrap overflow-auto flex-1 custom-scrollbar leading-relaxed">{details.input}</pre>
                  </div>
                  <div className="glass-strong rounded-lg p-5 border-l-4 border-l-purple-500 flex flex-col bg-zinc-900/60 shadow-lg">
                    <h5 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2">{s.desc} - Output / Result</h5>
                    <pre className="font-mono text-[11px] text-zinc-300 whitespace-pre-wrap overflow-auto flex-1 custom-scrollbar leading-relaxed">{details.output}</pre>
                  </div>
                </div>
              );
            })}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
