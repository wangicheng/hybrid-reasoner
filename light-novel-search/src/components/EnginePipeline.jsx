import { useState, useEffect } from "react";
import { cn } from "../lib/utils";
import { motion, AnimatePresence } from "framer-motion";

export function EnginePipeline({ currentStep, engineData }) {
  const [selectedStep, setSelectedStep] = useState(1);

  // Auto-select the current active step as it progresses
  useEffect(() => {
    if (currentStep > 0 && currentStep <= 4) {
      setSelectedStep(currentStep);
    }
  }, [currentStep]);

  const steps = [
    {
      id: 1,
      title: "查詢規劃器",
      desc: "SoT 並行解析層",
      nodes: ["語意理解", "標籤投影", "結構過濾"],
    },
    {
      id: 2,
      title: "檢索核心",
      desc: "雙路徑檢索",
      nodes: ["向量資料庫", "屬性資料庫"],
    },
    {
      id: 3,
      title: "後置處理",
      desc: "評分與融合",
      nodes: ["規則過濾層"],
    },
    {
      id: 4,
      title: "精準重排序",
      desc: "Rerank",
      nodes: ["最終排序"],
    },
  ];

  const getDetails = (stepId) => {
    if (!engineData) return { input: "等待引擎回應中...", output: "處理中..." };

    switch (stepId) {
      case 1:
        return {
          input: `> 系統提示詞 (System Prompt)：\n請解析使用者意圖並輸出 JSON。\n> 使用者輸入：\n「${engineData.query}」`,
          output: JSON.stringify({
            search_terms: engineData.search_terms,
            generated_keywords: engineData.generated_keywords,
            tags: engineData.tag_intent?.positive_terms || [],
            negative_tags: engineData.tag_intent?.negative_terms || []
          }, null, 2)
        };
      case 2:
        return {
          input: `> 向量查詢 (Vector Query)：\n[ 語意嵌入轉換: "${engineData.search_terms}" ]\n> 屬性查詢 (Attribute Query)：\n強制標籤過濾與匹配，包含: [${(engineData.tag_intent?.positive_terms || []).join(", ")}]`,
          output: `> 執行結果：\n已從底層向量資料庫召回，並取得候選名單，準備進行後置處理。`
        };
      case 3:
        return {
          input: `> 融合參數：\nVector Score + Semantic Attributes\n> 規則引擎執行目標：\n套用結構化條件過濾，排除: [${(engineData.tag_intent?.negative_terms || []).join(", ")}]`,
          output: `> 過濾報告：\n成功過濾不符條件的書籍。\n目前剩餘有效候選集：${engineData.results?.length || 0} 筆，準備進行精準重排序。`
        };
      case 4:
        return {
          input: `> 重排序模型 (Reranker)：\n將使用者完整 Query 與候選小說進行深層注意力機制交互運算...`,
          output: `> 排序完成：\n選出最終結果\n` + 
            (engineData.results?.slice(0, 3).map((r, i) => `${i+1}. ${r.item?.name || "未知"} (評分: ${Math.round(r.score*100)}%)`).join("\n") || "")
        };
      default:
        return { input: "", output: "" };
    }
  };

  return (
    <div className="w-full py-8 mt-4 overflow-hidden hidden md:block">
      <div className="flex items-start justify-between max-w-5xl mx-auto relative">
        {/* Connecting Line */}
        <div className="absolute top-[3.5rem] left-0 w-full h-1 bg-zinc-800/50 rounded-full z-0">
          <motion.div
            className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 rounded-full"
            initial={{ width: "0%" }}
            animate={{
              width: currentStep === 0 ? "0%" : `${((Math.min(currentStep, 4) - 0.5) / 4) * 100}%`,
            }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
          />
        </div>

        {steps.map((step) => {
          const isActive = currentStep === step.id;
          const isPast = currentStep > step.id;
          const isPending = currentStep < step.id;
          const isSelected = selectedStep === step.id;

          return (
            <div 
              key={step.id} 
              className="relative z-10 flex flex-col items-center flex-1 cursor-pointer group"
              onClick={() => setSelectedStep(step.id)}
            >
              {/* Step Icon / Node */}
              <motion.div
                className={cn(
                  "w-14 h-14 rounded-xl flex items-center justify-center border-2 transition-all duration-500 mb-4 bg-void relative",
                  isActive
                    ? "border-cyan-400 shadow-[0_0_20px_rgba(0,240,255,0.4)] text-cyan-400"
                    : isPast
                    ? "border-purple-500 text-purple-400"
                    : "border-zinc-700 text-zinc-600",
                  isSelected && !isActive && "ring-2 ring-white/20 ring-offset-2 ring-offset-void"
                )}
                animate={isActive ? { scale: [1, 1.1, 1] } : { scale: 1 }}
                transition={{ repeat: isActive ? Infinity : 0, duration: 2 }}
              >
                <span className="font-mono text-xl font-bold">{step.id}</span>
                {isSelected && (
                  <div className="absolute -bottom-3 w-2 h-2 rounded-full bg-white animate-pulse" />
                )}
              </motion.div>

              {/* Step Title */}
              <div className="text-center">
                <h4
                  className={cn(
                    "font-bold text-lg transition-colors duration-500 group-hover:text-white",
                    isActive ? "text-cyan-400" : isPast ? "text-purple-400" : "text-zinc-500"
                  )}
                >
                  {step.title}
                </h4>
                <p className="text-xs text-zinc-500 mb-3">{step.desc}</p>
              </div>

              {/* Sub-nodes */}
              <div className="flex flex-col gap-2 mt-2 items-center">
                {step.nodes.map((node, i) => (
                  <motion.div
                    key={i}
                    className={cn(
                      "px-3 py-1.5 rounded-md text-xs font-medium border backdrop-blur-sm transition-all duration-500",
                      isActive
                        ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                        : isPast
                        ? "border-purple-500/20 bg-purple-500/5 text-purple-300"
                        : "border-zinc-800 bg-zinc-900/50 text-zinc-600"
                    )}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{
                      opacity: isPending ? 0.3 : 1,
                      y: 0,
                    }}
                    transition={{ delay: i * 0.1 }}
                  >
                    {node}
                  </motion.div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Details Panel */}
      <div className="mt-12 max-w-5xl mx-auto h-48 relative">
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
                  <div className="glass-strong rounded-lg p-5 border-l-4 border-l-cyan-500 flex flex-col">
                    <h5 className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">Input / Task</h5>
                    <pre className="font-mono text-xs text-zinc-300 whitespace-pre-wrap overflow-auto flex-1">{details.input}</pre>
                  </div>
                  <div className="glass-strong rounded-lg p-5 border-l-4 border-l-purple-500 flex flex-col">
                    <h5 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2">Output / Result</h5>
                    <pre className="font-mono text-xs text-zinc-300 whitespace-pre-wrap overflow-auto flex-1">{details.output}</pre>
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
