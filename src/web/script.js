const API_URL = "/api/search";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function performSearch() {
  const query = document.getElementById("query-input").value;
  const selectedModelId = document.getElementById("model-select").value;
  const customModelId = document.getElementById("custom-model-input").value.trim();
  const modelId = customModelId || selectedModelId;
  const resultsContainer = document.getElementById("results-container");
  const interpretationBox = document.getElementById("search-interpretation");
  const loading = document.getElementById("loading");

  if (!query) return;

  resultsContainer.innerHTML = "";
  interpretationBox.innerHTML = "";
  interpretationBox.classList.add("hidden");
  loading.classList.remove("hidden");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        model_id: modelId,
      }),
    });

    const data = await response.json();
    loading.classList.add("hidden");

    if (!response.ok) {
      throw new Error(data.detail || "搜尋發生錯誤");
    }

    if (data.parsed_criteria && data.parsed_criteria.length > 0) {
      displayInterpretation(interpretationBox, data.parsed_criteria, data.query_vector, data);
    }

    if (data.error) {
      const errorNotice = document.createElement("div");
      errorNotice.style.cssText =
        "color:#d32f2f;background:#ffebee;padding:15px;border-radius:8px;margin-bottom:20px;text-align:center;";
      errorNotice.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> 檢索過程發生部分錯誤：${escapeHtml(data.error)}`;
      resultsContainer.appendChild(errorNotice);
    }

    if (data.is_relaxed) {
      const relaxedNotice = document.createElement("div");
      relaxedNotice.className = "relaxed-notice";
      relaxedNotice.innerHTML = `
        <i class="fa-solid fa-lightbulb"></i>
        <span>系統已自動放寬部分條件，讓結果更容易命中。</span>
      `;
      resultsContainer.appendChild(relaxedNotice);
    }

    if (data.results && data.results.length > 0) {
      data.results.forEach(result => {
        resultsContainer.appendChild(createResultCard(result));
      });
    } else {
      resultsContainer.innerHTML = '<div style="text-align:center; padding:20px;">找不到符合條件的結果</div>';
    }
  } catch (error) {
    console.error("Error:", error);
    loading.classList.add("hidden");
    resultsContainer.innerHTML =
      `<div style="color:#d32f2f;background:#ffebee;padding:15px;border-radius:8px;text-align:center;">` +
      `<i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(
        error.message || "發生錯誤，請檢查後端服務是否啟動"
      )}</div>`;
  }
}

function displayInterpretation(container, criteriaList, queryVector, data) {
  container.classList.remove("hidden");

  const semanticQueries = [];
  const filters = [];

  criteriaList.forEach(c => {
    if (c.name === "semantic_similarity") {
      const qt = c.parameters.query_text;
      if (qt) semanticQueries.push({ text: qt, is_negative: c.is_negative });
    } else if (c.name === "status_check") {
      filters.push(`狀態：${c.parameters.target_status}`);
    } else if (c.name === "author_match") {
      filters.push(`作者：${c.parameters.author_name}`);
    } else if (c.name === "numeric_range" && c.parameters.field === "words_total") {
      const minV = c.parameters.min_val ? Math.round(c.parameters.min_val / 10000) + " 萬" : null;
      const maxV = c.parameters.max_val ? Math.round(c.parameters.max_val / 10000) + " 萬" : null;
      if (minV && maxV) filters.push(`字數：${minV} ~ ${maxV}`);
      else if (minV) filters.push(`字數：≥ ${minV}`);
      else if (maxV) filters.push(`字數：≤ ${maxV}`);
    }
  });

  const searchTermsRaw = (data && data.search_terms) || "";
  const searchTerms = typeof searchTermsRaw === "string" ? (searchTermsRaw ? [searchTermsRaw] : []) : searchTermsRaw;
  const genKeywords = (data && data.generated_keywords) || [];
  const relatedBooks = (data && data.related_books) || [];
  const hypoIntro = (data && data.hypothetical_intro) || "";

  let html = `<h3><i class="fa-solid fa-robot"></i> AI 查詢解析</h3><div>`;

  if (searchTerms.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-magnifying-glass"></i> 主要查詢</strong> `;
    html += searchTerms.map(t => `<span class="criteria-tag">${escapeHtml(t)}</span>`).join("");
    html += `</div>`;
  }

  if (genKeywords.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-wand-magic-sparkles"></i> 擴展關鍵詞</strong> `;
    html += genKeywords
      .map(k => `<span class="criteria-tag" style="background:#e8f5e9;color:#2e7d32;">${escapeHtml(k)}</span>`)
      .join("");
    html += `</div>`;
  }

  const posSemantic = semanticQueries.filter(s => !s.is_negative);
  if (posSemantic.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-layer-group"></i> 語意條件</strong> `;
    html += posSemantic
      .map(s => `<span class="criteria-tag" style="background:#e3f2fd;color:#1565c0;">${escapeHtml(s.text)}</span>`)
      .join("");
    html += `</div>`;
  }

  const negSemantic = semanticQueries.filter(s => s.is_negative);
  if (negSemantic.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-ban"></i> 排除條件</strong> `;
    html += negSemantic
      .map(s => `<span class="criteria-tag" style="background:#ffebee;color:#c62828;">${escapeHtml(s.text)}</span>`)
      .join("");
    html += `</div>`;
  }

  if (filters.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-filter"></i> 硬性篩選</strong> `;
    html += filters
      .map(f => `<span class="criteria-tag" style="background:#fff3e0;color:#e65100;">${escapeHtml(f)}</span>`)
      .join("");
    html += `</div>`;
  }

  if (relatedBooks.length > 0) {
    html += `<div style="margin-bottom:10px;"><strong><i class="fa-solid fa-book"></i> 相關書籍</strong>`;
    html += relatedBooks
      .map(book => {
        const tags = Array.isArray(book.tags) ? book.tags.slice(0, 8) : [];
        const tagsHtml = tags
          .map(t => `<span class="criteria-tag" style="background:#fce4ec;color:#ad1457;">#${escapeHtml(t)}</span>`)
          .join("");
        const metaParts = [book.author, book.classification, book.publish_status].filter(Boolean).map(escapeHtml);
        const meta = metaParts.length > 0 ? metaParts.join(" · ") : "";
        const intro = book.intro ? escapeHtml(book.intro) : "";
        const matchInfo =
          book.match_source || book.match_score !== undefined
            ? `<div style="font-size:0.8em;color:#777;margin-top:4px;">${escapeHtml(
                book.match_source || ""
              )}${book.match_score !== undefined ? ` · ${Number(book.match_score).toFixed(3)}` : ""}</div>`
            : "";

        return `
          <div style="margin-top:8px;padding:10px 12px;border:1px solid #f3d3df;border-radius:8px;background:#fff8fb;">
            <div style="font-weight:600;">${escapeHtml(book.name || "")}</div>
            ${meta ? `<div style="font-size:0.85em;color:#666;margin-top:2px;">${meta}</div>` : ""}
            ${tagsHtml ? `<div style="margin-top:6px;">${tagsHtml}</div>` : ""}
            ${intro ? `<div style="font-size:0.85em;color:#444;margin-top:6px;line-height:1.5;">${intro}</div>` : ""}
            ${matchInfo}
          </div>
        `;
      })
      .join("");
    html += `</div>`;
  }

  if (hypoIntro) {
    html += `
      <div style="margin-top:8px;">
        <div onclick="togglePayload(this)" style="cursor:pointer;color:var(--primary-color);font-size:0.85em;display:inline-block;">
          <i class="fa-solid fa-file-lines"></i> <span class="toggle-text">展開 HyDE 預想內容</span>
        </div>
        <div class="raw-payload hidden" style="background:#f8f4ff;border-left:3px solid #7c4dff;padding:10px 14px;border-radius:0 6px 6px 0;font-size:0.85em;color:#333;margin-top:8px;">${escapeHtml(
          hypoIntro
        )}</div>
      </div>`;
  }

  if (queryVector && Array.isArray(queryVector) && queryVector.length > 0) {
    const dim = queryVector.length;
    const preview = queryVector.slice(0, 5).map(v => Number(v).toFixed(3)).join(", ");
    html += `<div style="margin-top:8px;padding-top:8px;border-top:1px dashed #ccc;font-size:0.85em;color:#666;">`;
    html += `<strong><i class="fa-solid fa-location-crosshairs"></i> 查詢向量</strong> [${preview}, ...] <span style="background:#eee;padding:2px 6px;border-radius:4px;font-size:0.8em;">${dim} 維</span>`;
    html += `</div>`;
  }

  html += `</div>`;

  const payloadJson = JSON.stringify(criteriaList, null, 2);
  html += `
    <div style="margin-top:12px;border-top:1px dashed #ccc;padding-top:10px;">
      <div onclick="togglePayload(this)" style="cursor:pointer;color:var(--primary-color);font-size:0.85em;display:inline-block;">
        <i class="fa-solid fa-code"></i> <span class="toggle-text">展開查詢解析 JSON</span>
      </div>
      <pre class="raw-payload hidden" style="background:#282c34;color:#abb2bf;padding:12px;border-radius:6px;font-size:0.85em;overflow-x:auto;margin-top:10px;max-height:300px;">${escapeHtml(payloadJson)}</pre>
    </div>
  `;

  container.innerHTML = html;
}

window.togglePayload = function (element) {
  const pre = element.nextElementSibling;
  const textSpan = element.querySelector(".toggle-text");
  if (pre.classList.contains("hidden")) {
    pre.classList.remove("hidden");
    textSpan.innerText = "收起內容";
  } else {
    pre.classList.add("hidden");
    textSpan.innerText = "展開內容";
  }
};

function createResultCard(result) {
  const item = result.item;
  const card = document.createElement("div");
  card.className = "result-card";
  const bookId = String(item.id || item.book_id || "");
  const isInBookshelf = window.Bookshelf ? window.Bookshelf.isInBookshelf(item) : false;
  const canSaveToBookshelf = Boolean(bookId);

  const tagsHtml = (item.tags || []).map(tag => `<span class="tag">#${escapeHtml(tag)}</span>`).join("");

  let breakdownHtml = '<div class="score-breakdown">';
  if (result.breakdown) {
    result.breakdown.forEach(b => {
      const rawScore = typeof b.raw_score === "number" ? b.raw_score : Number(b.raw_score || 0);
      const weightedScore = b.weighted_score !== undefined ? Number(b.weighted_score || 0) : Number(rawScore || 0);
      const scoreText = Number(weightedScore).toFixed(3);
      const widthPercent = Math.max(0, Math.min(weightedScore * 100, 100));
      const label = escapeHtml(b.label || b.criteria);
      const reasonTextRaw = b.reason ? String(b.reason) : "";
      const isTagOrKeyword = (b.label || "").includes("標籤") || (b.label || "").includes("關鍵");
      const hideReason = isTagOrKeyword || b.criteria === "semantic_similarity";
      const reasonText = hideReason ? "" : escapeHtml(reasonTextRaw);

      breakdownHtml += `
        <div class="score-bar-container">
          <span class="score-label">${label}</span>
          <span class="score-value">${scoreText}</span>
          <div class="progress-track">
            <div class="progress-fill" style="width:${widthPercent}%; background-color:${getColorForCriteria(b.criteria)}"></div>
          </div>
        </div>
        ${reasonText ? `<div class="score-reason">${reasonText}</div>` : ""}
      `;
    });
  }
  breakdownHtml += "</div>";

  const scoreBadgeValue = Number(result.score || 0).toFixed(4);

  let explanationHtml = "";
  if (result.explanation) {
    const criteriaCount = (result.breakdown || []).length;
    explanationHtml = `
      <div class="ai-explanation-box">
        <div class="ai-header" onclick="toggleExplanation(this)">
          <span><i class="fa-solid fa-robot"></i> AI 解釋</span>
          <span style="font-size:0.8em; color:#666; margin-left:10px;">(${criteriaCount} 個判斷依據)</span>
          <i class="fa-solid fa-chevron-down" style="margin-left:auto;"></i>
        </div>
        <div class="ai-content hidden">
          ${result.explanation}
        </div>
      </div>
    `;
  }

  card.innerHTML = `
    <div class="card-header">
      <div>
        <h2 class="book-title">${escapeHtml(item.name || "")}</h2>
        <div class="book-meta">
          <i class="fa-solid fa-user-pen"></i> ${escapeHtml(item.author || "Unknown")} |
          <i class="fa-solid fa-book"></i> ${escapeHtml(item.classification || "未知分類")} |
          <i class="fa-solid fa-pen-nib"></i> ${item.words_total ? Number(item.words_total).toLocaleString() : "未知"} 字
        </div>
      </div>
      <div class="card-actions">
        <button type="button" class="bookshelf-toggle-btn ${isInBookshelf ? "saved" : ""}" data-book-id="${bookId}" ${
    canSaveToBookshelf ? "" : "disabled"
  }>
          <i class="fa-solid fa-bookmark"></i> ${canSaveToBookshelf ? (isInBookshelf ? "已加入書櫃" : "加入書櫃") : "無法收藏"}
        </button>
        <div class="total-score">Score: ${scoreBadgeValue}</div>
      </div>
    </div>

    <div class="tags">${tagsHtml}</div>
    <div class="intro">${escapeHtml(item.intro || "無簡介")}</div>

    ${breakdownHtml}
    ${explanationHtml}
  `;

  const toggleButton = card.querySelector(".bookshelf-toggle-btn");
  if (toggleButton && window.Bookshelf && canSaveToBookshelf) {
    toggleButton.addEventListener("click", function () {
      const saved = window.Bookshelf.toggleBook(item);
      setBookshelfButtonState(toggleButton, saved);
    });
  }

  return card;
}

function setBookshelfButtonState(button, saved) {
  if (!button) return;
  button.classList.toggle("saved", saved);
  button.innerHTML = `<i class="fa-solid fa-bookmark"></i> ${saved ? "已加入書櫃" : "加入書櫃"}`;
}

function syncBookshelfButtons() {
  if (!window.Bookshelf) return;
  const buttons = document.querySelectorAll(".bookshelf-toggle-btn[data-book-id]");
  buttons.forEach(button => {
    const bookId = button.getAttribute("data-book-id");
    const saved = bookId ? window.Bookshelf.hasBookId(bookId) : false;
    setBookshelfButtonState(button, saved);
  });
}

window.toggleExplanation = function (headerElement) {
  const content = headerElement.nextElementSibling;
  const icon = headerElement.querySelector(".fa-chevron-down") || headerElement.querySelector(".fa-chevron-up");

  if (content.classList.contains("hidden")) {
    content.classList.remove("hidden");
    if (icon) {
      icon.classList.remove("fa-chevron-down");
      icon.classList.add("fa-chevron-up");
    }
  } else {
    content.classList.add("hidden");
    if (icon) {
      icon.classList.remove("fa-chevron-up");
      icon.classList.add("fa-chevron-down");
    }
  }
};

function getColorForCriteria(criteriaName) {
  switch (criteriaName) {
    case "semantic_similarity":
      return "#4a90e2";
    case "numeric_range":
      return "#66bb6a";
    case "keyword_match":
      return "#9c27b0";
    case "status_check":
      return "#ff9800";
    case "author_match":
      return "#00bcd4";
    default:
      return "#ab47bc";
  }
}

document.getElementById("query-input").addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    performSearch();
  }
});

document.addEventListener("bookshelf:changed", function () {
  syncBookshelfButtons();
});
