import { useState, useEffect, useRef } from "react";
import { cn } from "../lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import Tilt from "react-parallax-tilt";

const dedupeDisplayTerms = (terms = [], excludeTerms = []) => {
  const seen = new Set(
    excludeTerms
      .map(term => String(term || "").replace(/\s+/g, "").trim())
      .filter(Boolean)
  );

  return terms.reduce((acc, term) => {
    const value = String(term || "").trim();
    const normalised = value.replace(/\s+/g, "");
    if (!normalised || seen.has(normalised)) {
      return acc;
    }

    seen.add(normalised);
    acc.push(value);
    return acc;
  }, []);
};

export function EnginePipeline({ currentStep, engineData, results }) {
  const [selectedStep, setSelectedStep] = useState(1);
  const [selectedBranch, setSelectedBranch] = useState("semantic"); // "semantic" | "structure" | "tag"
  const [activeBook, setActiveBook] = useState(null);
  const wrapperRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleCollapse = () => {
    setIsCollapsed(true);
    setSelectedStep(7);
  };

  // Auto-select the current active step as it progresses
  useEffect(() => {
    if (!isCollapsed && currentStep > 0 && currentStep <= 7) {
      setSelectedStep(currentStep);
      if (currentStep === 2) {
        setSelectedBranch("semantic");
      }
    }
  }, [currentStep, isCollapsed]);

  // Responsive Scaling logic using ResizeObserver
  useEffect(() => {
    if (!wrapperRef.current) return;
    const element = wrapperRef.current;

    const observer = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const parentWidth = entry.contentRect.width;
        if (parentWidth > 0) {
          const newScale = Math.min(1, parentWidth / 1240);
          setScale(newScale);
        }
      }
    });

    if (element.parentElement) {
      observer.observe(element.parentElement);
    }

    return () => observer.disconnect();
  }, []);

  const steps = [
    { id: 1, title: "輸入階段", desc: "查詢 (Query)" },
    { id: 2, title: "並行解析層", desc: "語意理解分支" },
    { id: 3, title: "並行解析層", desc: "結構與標籤分支" },
    { id: 4, title: "需求合併", desc: "Requirement Merging" },
    { id: 5, title: "檢索與過濾", desc: "外部資料庫篩選" },
    { id: 6, title: "後處理與輸出", desc: "精確排序與輸出" },
    { id: 7, title: "推薦結果", desc: "全息卡片" }
  ];

  const getDetails = (stepId) => {
    if (!engineData) return { input: "等待引擎回應中...", output: "處理中..." };

    const semanticConcepts = dedupeDisplayTerms(
      engineData.parsed_criteria
        ?.filter(c => c.name === "semantic_similarity")
        ?.map(c => c.parameters?.query_text) || [],
      [engineData.search_terms]
    );
    const positiveTerms = dedupeDisplayTerms(engineData.tag_intent?.positive_terms || []);
    const negativeTerms = dedupeDisplayTerms(engineData.tag_intent?.negative_terms || []);
    const fuzzyPositiveTerms = dedupeDisplayTerms(engineData.tag_intent?.fuzzy_positive_terms || []);
    const fuzzyNegativeTerms = dedupeDisplayTerms(engineData.tag_intent?.fuzzy_negative_terms || []);
    const displaySearchTerms = dedupeDisplayTerms(String(engineData.search_terms || "").split(/\s+/g)).join(" ");

    // 💡 前置擋板：如果當前引擎還沒執行到該步驟，顯示等待中的專業提示，防止點擊時顯示「0筆」等不準確資訊
    if (currentStep < stepId) {
      switch (stepId) {
        case 2:
        case 3:
          return {
            input: `> 並行解析層 (Parallel Parsing Layer)：\n利用大語言模型對查詢進行核心語意抽取，並並行進行結構化硬過濾屬性判定與 Whitelist 標籤投影映射...\n\n> 任務狀態：正在等待輸入階段完成並發送 API 請求。`,
            output: `> 輸出結果：\n並行解析尚未開始...`
          };
        case 4:
          return {
            input: `> 需求合併 (Merging Phase)：\n彙整來自語意分支的精煉文本、結構化欄位過濾器、以及標籤映射權重，編譯出統一的檢索需求 payload...\n\n> 任務狀態：正在等待並行解析分支處理完成。`,
            output: `> 輸出結果：\n需求合併尚未開始...`
          };
        case 5:
          return {
            input: `> 檢索與過濾階段 (Retrieval & Filter Phase)：\n將語意嵌入發送至 Qdrant 向量資料庫做召回並套用物理過濾，同時以 BM25 在 SQLite 中做詞頻檢索，並實施雙軌評分融合 (Fusion) 與 Graceful Degradation (放寬降權) 救援機制...\n\n> 任務狀態：正在等待解析與需求合併完成。`,
            output: `> 輸出結果：\n雙路徑資料庫檢索與規則過濾尚未開始...`
          };
        case 6:
          return {
            input: `> LLM 重排序階段 (Rerank Phase)：\n使用 Cross-Attention 交叉注意力機制，將資料庫召回的 top-N 候選小說與使用者完整查詢傳入 Reranker 大模型，透過 PermSC (排列自我一致性) 進行深度重排...\n\n> 任務狀態：正在等待資料庫檢索與過濾輸出候選名單。`,
            output: `> 輸出結果：\nLLM 重排序與最終計分尚未開始...`
          };
        default:
          return { input: "> 狀態：等待前面步驟完成...", output: "> 尚未開始..." };
      }
    }

    switch (stepId) {
      case 1:
        return {
          input: `> 系統就緒：\n準備接收使用者輸入並開始處理。\n> 使用者輸入：\n「${engineData.query || ""}」`,
          output: `> 階段完成：\n已進入查詢管道，準備進行並行解析。\n原始輸入字串已快取。`
        };
      case 2:
        if (!engineData.parsed_criteria && !engineData.tag_intent) {
          return {
            input: `> 語意理解分支任務 (Semantic Understanding Pass)：\n分析查詢的核心語意...\n\n> 任務狀態：執行中...`,
            output: `> 輸出結果：\n正在進行語意理解 LLM 推理，請稍候...`
          };
        }
        return {
          input: `> 語意理解分支任務 (Semantic Understanding Pass)：\n使用大型語言模型分析查詢的核心語意，提取高階概念與意圖，並排除非關鍵的修飾詞，生成精煉的檢索語意文本。\n\n> 任務狀態：已完成\n> 輸入查詢：\n「${engineData.query || ""}」`,
          output: JSON.stringify({
            semantic_query_text: displaySearchTerms || engineData.search_terms || "",
            positive_concepts: semanticConcepts,
            negative_concepts: dedupeDisplayTerms(
              engineData.parsed_criteria
                ?.filter(c => c.name === "semantic_similarity" && c.is_negative)
                ?.map(c => c.parameters?.query_text) || []
            )
          }, null, 2)
        };
      case 3:
        if (!engineData.parsed_criteria && !engineData.tag_intent) {
          return {
            input: `> 結構限制與標籤投影任務...\n\n> 任務狀態：等待語意理解完成...`,
            output: `> 輸出結果：\n尚未開始...`
          };
        }
        if (selectedBranch === "structure") {
          return {
            input: `> 結構過濾分支任務 (Structured Constraints Pass)：\n分析查詢中的硬性約束條件（完結狀態、作者名稱、字數區間等），將其轉化為精確的資料庫屬性過濾欄位。\n\n> 任務狀態：已完成\n> 輸入資料：\n- 原始查詢：\n  「${engineData.query}」\n- 語意理解輸出 (引導輸入)：\n  * 精煉語意："${engineData.search_terms || "讀取中"}"\n  * 正向特徵概念：[${semanticConcepts.join(", ")}]`,
            output: JSON.stringify({
              status_filter: engineData.parsed_criteria?.find(c => c.name === "status_check")?.parameters?.target_status || "無限制",
              author_filter: engineData.parsed_criteria?.find(c => c.name === "author_match")?.parameters?.author_name || "無限制",
              word_count_filter: (() => {
                const numRange = engineData.parsed_criteria?.find(c => c.name === "numeric_range" && c.parameters?.field === "words_total");
                if (!numRange) return "無限制";
                return {
                  min_words: numRange.parameters?.min_val || 0,
                  max_words: numRange.parameters?.max_val || "無限制"
                };
              })()
            }, null, 2)
          };
        } else {
          return {
            input: `> 標籤投影分支任務 (Tag Projection Pass)：\n將查詢中提及的主題特徵、流派、元素等，映射至標準 Whitelist 小說標籤庫，並決定是否以 Exact 或 Fuzzy 進行匹配。\n\n> 任務狀態：已完成\n> 輸入資料：\n- 原始查詢：\n  「${engineData.query}」\n- 語意理解輸出 (引導輸入)：\n  * 精煉語意："${engineData.search_terms || "讀取中"}"\n  * 正向特徵概念：[${semanticConcepts.join(", ")}]`,
            output: JSON.stringify({
              positive_terms: positiveTerms,
              negative_terms: negativeTerms,
              fuzzy_positive_terms: fuzzyPositiveTerms,
              fuzzy_negative_terms: fuzzyNegativeTerms,
              tag_mappings: engineData.tag_mapping?.map(m => ({
                term: m.term,
                is_exact: m.is_exact,
                matched_tags: m.mappings
                  ?.filter(x => x.raw_score >= 0.7)
                  ?.slice(0, 3)
                  ?.map(x => `${x.tag} (相似度: ${Math.round(x.raw_score * 100)}%, 權重: ${x.scaled_score.toFixed(2)})`)
              })) || []
            }, null, 2)
          };
        }
      case 4:
        if (!engineData.search_terms) {
          return {
            input: `> 準備彙整下列分支結果：\n1. 語意脈絡\n2. 結構化限制\n3. 目標標籤集合`,
            output: `> 統一檢索需求：\n正在等待並行解析分支完成...`
          };
        }
        return {
          input: `> 準備彙整下列分支結果：\n1. 語意脈絡\n2. 結構化限制\n3. 目標標籤集合`,
          output: `> 統一檢索需求 (Merged Retrieval Query)：\n` + JSON.stringify({
            search_terms: displaySearchTerms || engineData.search_terms,
            generated_keywords: engineData.generated_keywords || [],
            tags: positiveTerms,
            negative_tags: negativeTerms
          }, null, 2)
        };
      case 5:
        return {
          input: `> 雙路徑檢索與規則過濾 (Retrieval & Filtering):\n[ 語意嵌入轉換: "${displaySearchTerms || "語意載入中"}" ]\n> 強制過濾標籤: [${positiveTerms.join(", ")}]\n> 排除過濾標籤: [${negativeTerms.join(", ")}]`,
          output: `> 檢索結果：\n已從 Qdrant 向量資料庫與 SQLite 召回 ${engineData.candidate_count || 0} 筆原始候選作品。` +
            (engineData.filtered_count !== undefined
              ? `\n\n> 評分融合與規則過濾完成：\n經雙軌融合與硬過濾判定，篩選出 ${engineData.filtered_count} 筆符合條件作品準備進行重排。`
              : `\n\n> 狀態：正在進行規則過濾與雙軌評分融合 (Fusion) 計算...`)
        };
      case 6:
        return {
          input: `> 大語言模型重排序 (LLM Reranker):\n將符合條件的小說與使用者原始查詢送入大模型，以排列自我一致性 (PermSC) 進行 Cross-Attention 交叉注意力重排序，精準判定契合度...`,
          output: engineData.top_results
            ? `> LLM 重排序與最終計分完成！\n\n> Top 3 最終推薦順序：\n` +
            engineData.top_results.map((r, i) => `${i + 1}. ${r.name || "未知"} (契合分數: ${Math.round(r.score * 100)})`).join("\n")
            : `> 狀態：正在調用大語言模型 Reranker，對候選作品進行精細化重排運算...`
        };
      default:
        return { input: "", output: "" };
    }
  };

  // Node geometries and X,Y coordinates
  const W_RECT = 130, H_RECT = 48;
  const D_DIAM = 64;
  const W_OVAL = 110, H_OVAL = 40;
  const W_RES = 100, H_RES = 48;

  const X_IN = 80;
  const X_SEM = 230;
  const X_BR = 440;
  const X_MRG = 600;
  const X_RUL = 740;
  const X_SCO = 890;
  const X_LLM = 1040;
  const X_RES = 1170;

  const Y_T = 100, Y_M = 220, Y_B = 340, Y_O = 60;

  // Path commands (M start L end)
  const lines = [
    { id: 'l1', d: `M 145 ${Y_M} L 165 ${Y_M}`, activeAt: 2 },
    { id: 'l2a', d: `M 295 ${Y_M} C 330 ${Y_M}, 340 ${Y_T}, 375 ${Y_T}`, activeAt: 3 },
    { id: 'l2c', d: `M 295 ${Y_M} C 330 ${Y_M}, 340 ${Y_B}, 375 ${Y_B}`, activeAt: 3 },
    { id: 'l3a', d: `M 505 ${Y_T} C 535 ${Y_T}, 545 ${Y_M}, 568 ${Y_M}`, activeAt: 4 },
    { id: 'l3b', d: `M 295 ${Y_M} L 568 ${Y_M}`, activeAt: 4 },
    { id: 'l3c', d: `M 505 ${Y_B} C 535 ${Y_B}, 545 ${Y_M}, 568 ${Y_M}`, activeAt: 4 },
    { id: 'l4', d: `M 632 ${Y_M} L 675 ${Y_M}`, activeAt: 5 },
    { id: 'l4a', d: `M ${X_RUL} 80 L ${X_RUL} 196`, activeAt: 5 },
    { id: 'l5', d: `M 805 ${Y_M} L 825 ${Y_M}`, activeAt: 5 },
    { id: 'l5a', d: `M ${X_SCO} 80 L ${X_SCO} 196`, activeAt: 5 },
    { id: 'l6', d: `M 955 ${Y_M} L 975 ${Y_M}`, activeAt: 6 },
    { id: 'l7', d: `M 1105 ${Y_M} L 1120 ${Y_M}`, activeAt: 7 },
  ];

  const nodeStyle = (stepId, isBranchSelected = false) => {
    const isActive = currentStep === stepId;
    const isPast = currentStep > stepId;
    const isPending = currentStep < stepId;
    const isSelected = stepId === 2 ? isBranchSelected : (selectedStep === stepId);

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
    <div className="w-full py-6 mt-4 hidden md:block relative group/pipeline">

      {/* Collapse Trigger (Floating elegant pill button) */}
      {!isCollapsed && (
        <div className="flex justify-end max-w-5xl mx-auto mb-4 px-2">
          <button
            onClick={handleCollapse}
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-900/60 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200 text-xs font-mono transition-all duration-300 shadow-md backdrop-blur-sm"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
            </svg>
            收合為簡潔進度條
          </button>
        </div>
      )}

      {!isCollapsed ? (
        <div
          ref={wrapperRef}
          className="w-full overflow-hidden flex justify-center animate-in fade-in slide-in-from-top-2 duration-500"
          style={{ height: `${400 * scale}px` }}
        >
          <div
            className="relative w-[1240px] h-[400px] shrink-0 select-none origin-top transition-transform duration-300 ease-out"
            style={{ transform: `scale(${scale})` }}
          >

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
              style={{ left: 150, top: 50, width: 370, height: 340 }}
              onClick={() => { setSelectedStep(2); setSelectedBranch("semantic"); }}
            >
              <div className="absolute -top-3 left-4 px-2 bg-void text-xs text-zinc-500 font-mono tracking-widest">
                PARALLEL PARSING LAYER
              </div>
              {(selectedStep === 2 || selectedStep === 3) && <div className="absolute -left-1 -bottom-2 w-2 h-2 rounded-full bg-white animate-pulse" />}
            </div>

            {/* Small Labels on paths */}
            <div className="absolute z-10 text-[10px] text-zinc-400 bg-void border border-zinc-800 px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap" style={{ left: 335, top: 120, transform: 'translate(-50%, -50%)' }}>
              經由語意脈絡引導
            </div>
            <div className="absolute z-10 text-[10px] text-zinc-400 bg-void border border-zinc-800 px-1.5 py-0.5 rounded shadow-sm whitespace-nowrap" style={{ left: 335, top: 310, transform: 'translate(-50%, -50%)' }}>
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
              className={cn(nodeStyle(3, selectedStep === 3 && selectedBranch === "structure"), "rounded-md")}
              style={{ left: X_BR, top: Y_T, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
              onClick={() => { setSelectedStep(3); setSelectedBranch("structure"); }}
            >
              結構過濾分支
            </div>
            <div
              className={cn(nodeStyle(2, selectedStep === 2 && selectedBranch === "semantic"), "rounded-md")}
              style={{ left: X_SEM, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
              onClick={() => { setSelectedStep(2); setSelectedBranch("semantic"); }}
            >
              語意理解分支
            </div>
            <div
              className={cn(nodeStyle(3, selectedStep === 3 && selectedBranch === "tag"), "rounded-md")}
              style={{ left: X_BR, top: Y_B, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
              onClick={() => { setSelectedStep(3); setSelectedBranch("tag"); }}
            >
              標籤投影分支
            </div>

            {/* 3. Merge Diamond */}
            <div
              className={nodeStyle(4)}
              style={{ left: X_MRG, top: Y_M, width: D_DIAM, height: D_DIAM, transform: 'translate(-50%, -50%) rotate(45deg)' }}
              onClick={() => setSelectedStep(4)}
            >
              <div
                style={{ transform: 'rotate(-45deg)' }}
                className="text-[11px] text-center leading-tight font-bold whitespace-nowrap"
              >
                需求合併
              </div>
            </div>

            {/* 4. Filter & Retrieval Layer */}
            <div
              className={cn(nodeStyle(5), "rounded-full shadow-inner")}
              style={{ left: X_RUL, top: Y_O, width: W_OVAL, height: H_OVAL, transform: 'translate(-50%, -50%)' }}
              onClick={() => setSelectedStep(5)}
            >
              屬性資料庫
            </div>
            <div
              className={cn(nodeStyle(5), "rounded-md")}
              style={{ left: X_RUL, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
              onClick={() => setSelectedStep(5)}
            >
              規則過濾
            </div>

            <div
              className={cn(nodeStyle(5), "rounded-full shadow-inner")}
              style={{ left: X_SCO, top: Y_O, width: W_OVAL, height: H_OVAL, transform: 'translate(-50%, -50%)' }}
              onClick={() => setSelectedStep(5)}
            >
              向量資料庫
            </div>
            <div
              className={cn(nodeStyle(5), "rounded-md")}
              style={{ left: X_SCO, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
              onClick={() => setSelectedStep(5)}
            >
              評分與融合
            </div>

            {/* 5. Post Processing */}
            <div
              className={cn(nodeStyle(6), "rounded-md")}
              style={{ left: X_LLM, top: Y_M, width: W_RECT, height: H_RECT, transform: 'translate(-50%, -50%)' }}
              onClick={() => setSelectedStep(6)}
            >
              LLM 重排
            </div>
            <div
              className={cn(nodeStyle(7), "rounded border-zinc-600 bg-zinc-800")}
              style={{ left: X_RES, top: Y_M, width: W_RES, height: H_RES, transform: 'translate(-50%, -50%)' }}
              onClick={() => setSelectedStep(7)}
            >
              結果
            </div>
          </div>
        </div>
      ) : (
        /* Collapsed super clean progress bar */
        <div
          onClick={() => setIsCollapsed(false)}
          className="w-full max-w-5xl mx-auto my-6 px-2 cursor-pointer group animate-in fade-in duration-300"
          title="點擊展開完整流程圖"
        >
          <div className="relative w-full h-2 bg-zinc-950 rounded-full border border-white/5 overflow-hidden shadow-inner group-hover:border-cyan-500/40 transition-colors duration-300">
            <div
              className="h-full rounded-full bg-gradient-to-r from-purple-600 via-purple-500 to-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.6)] transition-all duration-1000 ease-out"
              style={{ width: `${Math.max(8, (currentStep / 7) * 100)}%` }}
            />
          </div>
          <div className="flex justify-center mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase flex items-center gap-1">
              <svg className="w-3.5 h-3.5 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
              點擊展開完整流程圖
            </span>
          </div>
        </div>
      )}

      {/* Details Panel */}
      <div className={cn("mt-8 max-w-5xl mx-auto relative transition-all duration-500 pb-12", [6, 7].includes(selectedStep) ? "min-h-[380px] h-auto" : "h-48")}>
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedStep + "-" + selectedBranch}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className={cn("w-full", [6, 7].includes(selectedStep) ? "relative" : "absolute inset-0")}
          >
            {steps.map(s => {
              if (s.id !== selectedStep) return null;

              if (s.id === 7) {
                return (
                  <div key={s.id} className="w-full">
                    <h4 className="text-sm font-bold text-purple-400 uppercase tracking-widest mb-6 text-center font-mono">
                      — 系統檢索完成 · 全息推薦結果 —
                    </h4>
                    {results && results.length > 0 ? (
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {results.flatMap((book, idx) => {
                          const items = [];
                          if (idx === 10) {
                            items.push(
                              <div key="more-candidates-divider" className="col-span-full py-6 flex items-center justify-center gap-4">
                                <div className="h-[1px] flex-grow bg-gradient-to-r from-transparent via-zinc-700 to-transparent"></div>
                                <span className="text-xs font-mono tracking-widest text-zinc-500 uppercase">
                                  — 更多候選作品 (經由雙路徑召回) —
                                </span>
                                <div className="h-[1px] flex-grow bg-gradient-to-r from-transparent via-zinc-700 to-transparent"></div>
                              </div>
                            );
                          }
                          items.push(
                            <div
                              key={book.id}
                              className="animate-in fade-in zoom-in-95 duration-700 fill-mode-both"
                              style={{ animationDelay: `${idx * 80}ms` }}
                            >
                              <Tilt
                                tiltMaxAngleX={10}
                                tiltMaxAngleY={10}
                                perspective={1000}
                                scale={1.02}
                                transitionSpeed={2000}
                                className="h-full"
                              >
                                <div
                                  onClick={() => setActiveBook(book)}
                                  className={cn(
                                    "glass h-full rounded-2xl p-4 flex flex-col relative group overflow-hidden border transition-colors cursor-pointer",
                                    idx < 10
                                      ? "border-purple-500/20 hover:border-purple-500/60 shadow-[0_0_15px_rgba(168,85,247,0.05)]"
                                      : "border-white/5 hover:border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.02)]"
                                  )}
                                >
                                  {/* Cover Image Placeholder */}
                                  <div className="w-full h-48 bg-zinc-800 rounded-xl mb-4 overflow-hidden relative">
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent z-10"></div>
                                    <img src={book.cover} alt={book.title} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity duration-500 group-hover:scale-105" />

                                    {/* Top 10 Premium Badge */}
                                    {idx < 10 && (
                                      <div className="absolute top-3 right-3 z-20 px-2 py-0.5 rounded-full bg-purple-950/80 border border-purple-500/50 backdrop-blur-sm text-[10px] text-purple-300 font-mono tracking-wider font-bold">
                                        RERANK 精選 #{idx + 1}
                                      </div>
                                    )}
                                  </div>

                                  {/* Info */}
                                  <div className="flex-grow z-20">
                                    <h3 className={cn(
                                      "text-lg font-bold mb-2 transition-colors",
                                      idx < 10
                                        ? "text-white group-hover:text-purple-400"
                                        : "text-zinc-300 group-hover:text-cyan-400"
                                    )}>{book.title}</h3>
                                    <div className="flex flex-wrap gap-1.5">
                                      {(book.tags || []).slice(0, 3).map(t => (
                                        <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-zinc-300">
                                          {t}
                                        </span>
                                      ))}
                                      {book.tags && book.tags.length > 3 && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                                          +{book.tags.length - 3}
                                        </span>
                                      )}
                                    </div>
                                  </div>

                                  {/* Hover Glow Effect */}
                                  <div className={cn(
                                    "absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none",
                                    idx < 10
                                      ? "bg-gradient-to-br from-purple-500/10 via-transparent to-cyan-500/10"
                                      : "bg-gradient-to-br from-cyan-500/10 via-transparent to-purple-500/10"
                                  )}></div>
                                </div>
                              </Tilt>
                            </div>
                          );
                          return items;
                        })}
                      </div>
                    ) : (
                      <div className="glass-strong rounded-lg p-8 text-center text-zinc-400 font-mono">
                        &gt; 正在進行最後檢索與重排序，請稍候...
                      </div>
                    )}
                  </div>
                );
              }

              if (s.id === 6) {
                const details = getDetails(s.id);
                return (
                  <div key={s.id} className="flex flex-col h-full space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 shrink-0">
                      <div className="glass-strong rounded-lg p-5 border-l-4 border-l-cyan-500 flex flex-col bg-zinc-900/60 shadow-lg">
                        <h5 className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">{s.title} - Input / Task</h5>
                        <pre className="font-mono text-[11px] text-zinc-300 whitespace-pre-wrap overflow-auto h-24 custom-scrollbar leading-relaxed">{details.input}</pre>
                      </div>
                      <div className="glass-strong rounded-lg p-5 border-l-4 border-l-purple-500 flex flex-col bg-zinc-900/60 shadow-lg">
                        <h5 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2">{s.desc} - Output / Result</h5>
                        <pre className="font-mono text-[11px] text-zinc-300 whitespace-pre-wrap overflow-auto h-24 custom-scrollbar leading-relaxed">{details.output}</pre>
                      </div>
                    </div>
                    {/* Pre-Rerank Candidates Grid */}
                    <div className="flex-1 overflow-auto custom-scrollbar pt-2 pb-4 px-2">
                      <h4 className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-4 font-mono flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-yellow-500/70 animate-ping"></span>
                        正在等待大語言模型進行交叉重排序... (準備排序的候選池)
                      </h4>
                      {engineData.pre_rerank_candidates && engineData.pre_rerank_candidates.length > 0 ? (
                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
                          {engineData.pre_rerank_candidates.map((book, idx) => (
                            <div 
                              key={book.id || idx}
                              className="relative overflow-hidden rounded-xl border border-zinc-700/50 bg-zinc-900 overflow-hidden group grayscale opacity-70 animate-pulse aspect-[3/4]"
                            >
                              <img src={book.cover} alt={book.title || "未知"} className="w-full h-full object-cover opacity-40 bg-zinc-800" />
                              <div className="absolute inset-0 border-2 border-transparent transition-all duration-1000 group-hover:border-yellow-500/30"></div>
                              {/* Scanner Line */}
                              <div className="absolute left-0 right-0 h-1 bg-yellow-500/50 shadow-[0_0_8px_rgba(234,179,8,0.8)] filter blur-[1px] animate-scan"></div>
                              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-2 pt-6">
                                <p className="text-[10px] text-zinc-300 font-bold line-clamp-2 leading-tight">
                                  {book.title}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center text-zinc-600 font-mono text-sm py-8 border border-dashed border-zinc-800 rounded-lg">
                          尚未收到候選池資料...
                        </div>
                      )}
                    </div>
                  </div>
                );
              }

              const details = getDetails(s.id);

              let displayTitle = s.title;
              let displayDesc = s.desc;

              if (s.id === 2 || s.id === 3) {
                if (selectedBranch === "semantic") {
                  displayTitle = "並行解析層 (語意理解)";
                  displayDesc = "語意理解分支";
                } else if (selectedBranch === "structure") {
                  displayTitle = "並行解析層 (結構過濾)";
                  displayDesc = "結構過濾分支";
                } else if (selectedBranch === "tag") {
                  displayTitle = "並行解析層 (標籤投影)";
                  displayDesc = "標籤投影分支";
                }
              }

              return (
                <div key={s.id} className="grid grid-cols-2 gap-6 h-full">
                  <div className="glass-strong rounded-lg p-5 border-l-4 border-l-cyan-500 flex flex-col bg-zinc-900/60 shadow-lg">
                    <h5 className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">{displayTitle} - Input / Task</h5>
                    <pre className="font-mono text-[11px] text-zinc-300 whitespace-pre-wrap overflow-auto flex-1 custom-scrollbar leading-relaxed">{details.input}</pre>
                  </div>
                  <div className="glass-strong rounded-lg p-5 border-l-4 border-l-purple-500 flex flex-col bg-zinc-900/60 shadow-lg">
                    <h5 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2">{displayDesc} - Output / Result</h5>
                    <pre className="font-mono text-[11px] text-zinc-300 whitespace-pre-wrap overflow-auto flex-1 custom-scrollbar leading-relaxed">{details.output}</pre>
                  </div>
                </div>
              );
            })}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Novel Details Modal */}
      <AnimatePresence>
        {activeBook && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-void/90 backdrop-blur-md overflow-y-auto"
            onClick={() => setActiveBook(null)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="relative w-full max-w-4xl bg-zinc-900/95 border border-zinc-700/50 rounded-3xl overflow-hidden shadow-2xl p-6 md:p-8 flex flex-col md:flex-row gap-8 backdrop-blur-xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close Button */}
              <button
                onClick={() => setActiveBook(null)}
                className="absolute top-4 right-4 w-10 h-10 rounded-full flex items-center justify-center bg-white/5 border border-white/10 text-zinc-400 hover:text-white hover:bg-white/10 hover:border-white/20 transition-all z-30 animate-pulse"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>

              {/* Cover Column */}
              <div className="w-full md:w-1/3 flex flex-col items-center">
                <div className="w-full aspect-[3/4] max-w-[240px] rounded-2xl overflow-hidden border border-zinc-700/50 shadow-lg relative group">
                  <img
                    src={activeBook.cover}
                    alt={activeBook.title}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-zinc-950/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-4">
                    <span className="text-xs font-mono text-zinc-300">Holographic Card</span>
                  </div>
                </div>
              </div>

              {/* Details Column */}
              <div className="w-full md:w-2/3 flex flex-col justify-between overflow-y-auto max-h-[85vh] md:max-h-[500px] pr-2 custom-scrollbar">
                <div>
                  {/* Title & Author Info */}
                  <h2 className="text-2xl md:text-3xl font-extrabold text-white leading-tight mb-2 tracking-tight">
                    {activeBook.title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-3 mb-6">
                    <span className="px-3 py-1 rounded-full bg-zinc-800 text-zinc-300 text-xs font-medium border border-zinc-700">
                      作者：{activeBook.author}
                    </span>
                    <span className={cn(
                      "px-3 py-1 rounded-full text-xs font-mono font-bold border",
                      activeBook.status?.includes("完結") || activeBook.status?.toLowerCase().includes("completed")
                        ? "bg-purple-950/50 text-purple-300 border-purple-500/30"
                        : "bg-cyan-950/50 text-cyan-300 border-cyan-500/30"
                    )}>
                      狀態：{activeBook.status}
                    </span>
                    <span className="px-3 py-1 rounded-full bg-zinc-800 text-zinc-300 text-xs font-mono border border-zinc-700">
                      字數：{activeBook.words_total ? `${(activeBook.words_total / 10000).toFixed(1)}萬字` : "未統計"}
                    </span>
                  </div>

                  {/* Dynamic Tags */}
                  <div className="mb-6">
                    <h3 className="text-xs font-mono tracking-widest text-zinc-500 uppercase mb-2">標準標籤集合</h3>
                    <div className="flex flex-wrap gap-2">
                      {(activeBook.tags || []).map(t => (
                        <span key={t} className="text-xs px-2.5 py-1 rounded-lg bg-zinc-800/80 text-zinc-200 border border-zinc-700/50 font-medium hover:border-purple-400/50 hover:text-purple-300 transition-colors cursor-default">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Synopsis / Introduction */}
                  <div className="mb-6">
                    <h3 className="text-xs font-mono tracking-widest text-zinc-500 uppercase mb-2">小說簡介</h3>
                    <div className="bg-zinc-950/55 border border-zinc-800/80 rounded-2xl p-4 text-zinc-300 text-sm leading-relaxed whitespace-pre-wrap font-sans max-h-40 overflow-y-auto custom-scrollbar">
                      {activeBook.intro}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

