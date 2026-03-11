const API_URL = "/api/search";

async function performSearch() {
  const query = document.getElementById('query-input').value;
  const modelId = document.getElementById('model-select').value;
  const resultsContainer = document.getElementById('results-container');
  const interpretationBox = document.getElementById('search-interpretation');
  const loading = document.getElementById('loading');

  if (!query) return;

  // UI Reset
  resultsContainer.innerHTML = '';
  interpretationBox.innerHTML = '';
  interpretationBox.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        model_id: modelId
      })
    });

    const data = await response.json();
    loading.classList.add('hidden');

    if (!response.ok) {
      throw new Error(data.detail || '伺服器發生錯誤');
    }

    // --- Display Search Interpretation ---
    if (data.parsed_criteria && data.parsed_criteria.length > 0) {
      displayInterpretation(interpretationBox, data.parsed_criteria, data.query_vector, data);
    }

    if (data.error) {
      const errorNotice = document.createElement('div');
      errorNotice.style.cssText = "color: #d32f2f; background: #ffebee; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;";
      errorNotice.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> 檢索過程發生部分錯誤：${data.error}`;
      resultsContainer.appendChild(errorNotice);
    }

    // --- 延伸推薦提示 (Relaxed Search Notice) ---
    if (data.is_relaxed) {
      const relaxedNotice = document.createElement('div');
      relaxedNotice.className = 'relaxed-notice';
      relaxedNotice.innerHTML = `
        <i class="fa-solid fa-lightbulb"></i>
        <span>找不到完全符合篩選條件的書籍，已為您擴大搜尋範圍，以下為延伸推薦結果。</span>
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
    console.error('Error:', error);
    loading.classList.add('hidden');
    resultsContainer.innerHTML = `<div style="color:#d32f2f; background: #ffebee; padding: 15px; border-radius: 8px; text-align:center;"><i class="fa-solid fa-triangle-exclamation"></i> ${error.message || '發生錯誤，請檢查後端服務是否啟動'}</div>`;
  }
}

function displayInterpretation(container, criteriaList, queryVector, data) {
  container.classList.remove('hidden');

  // --- 從 criteria 中提取各類條件 ---
  const semanticQueries = [];
  const filters = [];

  criteriaList.forEach(c => {
    if (c.name === 'semantic_similarity') {
      const qt = c.parameters.query_text;
      if (qt) semanticQueries.push({ text: qt, is_negative: c.is_negative });
    } else if (c.name === 'status_check') {
      filters.push(`狀態: ${c.parameters.target_status}`);
    } else if (c.name === 'author_match') {
      filters.push(`作者: ${c.parameters.author_name}`);
    } else if (c.name === 'numeric_range' && c.parameters.field === 'words_total') {
      const minV = c.parameters.min_val ? Math.round(c.parameters.min_val / 10000) + '萬' : null;
      const maxV = c.parameters.max_val ? Math.round(c.parameters.max_val / 10000) + '萬' : null;
      if (minV && maxV) filters.push(`字數: ${minV}~${maxV}`);
      else if (minV) filters.push(`字數 ≥ ${minV}`);
      else if (maxV) filters.push(`字數 ≤ ${maxV}`);
    }
  });

  const searchTermsRaw = (data && data.search_terms) || "";
  const searchTerms = typeof searchTermsRaw === 'string' ? (searchTermsRaw ? [searchTermsRaw] : []) : searchTermsRaw;
  const genKeywords = (data && data.generated_keywords) || [];
  const refTags = (data && data.reference_tags) || [];
  const hypoIntro = (data && data.hypothetical_intro) || '';

  let html = `<h3><i class="fa-solid fa-robot"></i> AI 搜尋解析</h3><div>`;

  // 1. 搜尋關鍵字（LLM 從問題提取）
  if (searchTerms.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-magnifying-glass"></i> 提取關鍵字：</strong> `;
    html += searchTerms.map(t => `<span class="criteria-tag">${t}</span>`).join('');
    html += `</div>`;
  }

  // 2. LLM 擴展關鍵字
  if (genKeywords.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-wand-magic-sparkles"></i> 擴展關鍵字：</strong> `;
    html += genKeywords.map(k => `<span class="criteria-tag" style="background:#e8f5e9;color:#2e7d32;">${k}</span>`).join('');
    html += `</div>`;
  }

  // 3. 語意向量查詢（正向）
  const posSemantic = semanticQueries.filter(s => !s.is_negative);
  if (posSemantic.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-layer-group"></i> 語意向量查詢：</strong> `;
    html += posSemantic.map(s => `<span class="criteria-tag" style="background:#e3f2fd;color:#1565c0;">${s.text}</span>`).join('');
    html += `</div>`;
  }

  // 4. 排除條件（負向語意）
  const negSemantic = semanticQueries.filter(s => s.is_negative);
  if (negSemantic.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-ban"></i> 排除條件：</strong> `;
    html += negSemantic.map(s => `<span class="criteria-tag" style="background:#ffebee;color:#c62828;">${s.text}</span>`).join('');
    html += `</div>`;
  }

  // 5. 硬過濾器
  if (filters.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-filter"></i> 硬過濾器：</strong> `;
    html += filters.map(f => `<span class="criteria-tag" style="background:#fff3e0;color:#e65100;">${f}</span>`).join('');
    html += `</div>`;
  }

  // 5.5 參考書籍標籤
  if (refTags.length > 0) {
    html += `<div style="margin-bottom:6px;"><strong><i class="fa-solid fa-book"></i> 參考書籍標籤：</strong> `;
    html += refTags.map(t => `<span class="criteria-tag" style="background:#fce4ec;color:#ad1457;">#${t}</span>`).join('');
    html += `</div>`;
  }

  // 6. HyDE 假設簡介（折疊顯示）
  if (hypoIntro) {
    html += `
      <div style="margin-top:8px;">
        <div onclick="togglePayload(this)" style="cursor:pointer;color:var(--primary-color);font-size:0.85em;display:inline-block;">
          <i class="fa-solid fa-file-lines"></i> <span class="toggle-text">顯示 HyDE 假設簡介</span>
        </div>
        <div class="raw-payload hidden" style="background:#f8f4ff;border-left:3px solid #7c4dff;padding:10px 14px;border-radius:0 6px 6px 0;font-size:0.85em;color:#333;margin-top:8px;">${hypoIntro}</div>
      </div>`;
  }

  // 7. 向量座標
  if (queryVector && Array.isArray(queryVector) && queryVector.length > 0) {
    const dim = queryVector.length;
    const preview = queryVector.slice(0, 5).map(v => Number(v).toFixed(3)).join(', ');
    html += `<div style="margin-top:8px;padding-top:8px;border-top:1px dashed #ccc;font-size:0.85em;color:#666;">`;
    html += `<strong><i class="fa-solid fa-location-crosshairs"></i> 向量座標：</strong> [${preview}, ...] <span style="background:#eee;padding:2px 6px;border-radius:4px;font-size:0.8em;">${dim} 維度</span>`;
    html += `</div>`;
  }

  html += `</div>`;

  // 8. 原始 Payload JSON（折疊）
  const payloadJson = JSON.stringify(criteriaList, null, 2);
  html += `
    <div style="margin-top:12px;border-top:1px dashed #ccc;padding-top:10px;">
      <div onclick="togglePayload(this)" style="cursor:pointer;color:var(--primary-color);font-size:0.85em;display:inline-block;">
        <i class="fa-solid fa-code"></i> <span class="toggle-text">顯示查詢 Payload JSON</span>
      </div>
      <pre class="raw-payload hidden" style="background:#282c34;color:#abb2bf;padding:12px;border-radius:6px;font-size:0.85em;overflow-x:auto;margin-top:10px;max-height:300px;">${payloadJson}</pre>
    </div>
  `;

  container.innerHTML = html;
}

// 輔助函式：切換 Payload 顯示
window.togglePayload = function (element) {
  const pre = element.nextElementSibling;
  const textSpan = element.querySelector('.toggle-text');
  if (pre.classList.contains('hidden')) {
    pre.classList.remove('hidden');
    textSpan.innerText = '隱藏查詢 Payload JSON';
  } else {
    pre.classList.add('hidden');
    textSpan.innerText = '顯示查詢 Payload JSON';
  }
}

function createResultCard(result) {
  const item = result.item;
  const card = document.createElement('div');
  card.className = 'result-card';
  const bookId = String(item.id || item.book_id || '');
  const isInBookshelf = window.Bookshelf ? window.Bookshelf.isInBookshelf(item) : false;
  const canSaveToBookshelf = Boolean(bookId);

  // 1. 基本資訊與標籤
  const tagsHtml = (item.tags || []).map(tag => `<span class="tag">#${tag}</span>`).join('');

  // 2. 評分視覺化 (Breakdown Visualization)
  // 我們將每個評分項目的 weighted_score 繪製成條狀圖
  // 為了顯示比例，我們假設滿分是 1.0 (或根據實際總分動態調整)
  let breakdownHtml = '<div class="score-breakdown">';

  if (result.breakdown) {
    result.breakdown.forEach(b => {
      const originalScore = b.criteria === 'semantic_similarity'
        ? Number(b.raw_score || 0)
        : (b.normalized_score !== undefined ? Number(b.normalized_score) : Number(b.raw_score || 0));
      const rawScore = typeof b.raw_score === 'number' ? b.raw_score : Number(b.raw_score || 0);

      const weightedScore = b.weighted_score !== undefined ? Number(b.weighted_score || 0) : Number(originalScore || 0);
      const scoreText = Number(weightedScore).toFixed(3);

      // 如果分數小於 0，代表是扣分項（排除條件），我們用紅色顯示，絕對值做為寬度表示「懲罰力度」
      const isNegative = weightedScore < 0;
      const widthPercent = Math.max(0, Math.min(Math.abs(weightedScore) * 100, 100));
      const label = b.label || b.criteria;

      const reasonTextRaw = b.reason ? String(b.reason) : '';
      const isTagOrKeyword = label.includes('標籤') || label.includes('分類') || b.criteria === 'keyword_match';
      const hideReason = isTagOrKeyword || b.criteria === 'semantic_similarity' || reasonTextRaw.includes('未找到關鍵字');
      const reasonText = hideReason ? '' : reasonTextRaw;

      breakdownHtml += `
                <div class="score-bar-container">
                    <span class="score-label">${label}</span>
                    <span class="score-value">${scoreText}</span>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${widthPercent}%; background-color: ${isNegative ? '#ef5350' : getColorForCriteria(b.criteria)}"></div>
                    </div>
                </div>
                ${reasonText ? `<div class="score-reason">${reasonText}</div>` : ''}
            `;
    });
  }
  breakdownHtml += '</div>';

  const scoreBadgeLabel = 'Score';
  const scoreBadgeValue = Number(result.score || 0).toFixed(4);

  // 3. AI 解釋區塊 (Collapsible)
  // 只有當後端有回傳 explanation 時才顯示
  let explanationHtml = '';
  if (result.explanation) {
    const criteriaCount = (result.breakdown || []).length;
    explanationHtml = `
            <div class="ai-explanation-box">
                <div class="ai-header" onclick="toggleExplanation(this)">
                    <span><i class="fa-solid fa-robot"></i> AI 推薦理由</span>
                    <span style="font-size:0.8em; color:#666; margin-left:10px;">(基於 ${criteriaCount} 項評分指標分析)</span>
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
                <h2 class="book-title">${item.name}</h2>
                <div class="book-meta">
                    <i class="fa-solid fa-user-pen"></i> ${item.author} | 
                    <i class="fa-solid fa-book"></i> ${item.classification || '未分類'} | 
                    <i class="fa-solid fa-pen-nib"></i> ${item.words_total ? item.words_total.toLocaleString() : '未知'}字
                </div>
            </div>
                <div class="card-actions">
                  <button type="button" class="bookshelf-toggle-btn ${isInBookshelf ? 'saved' : ''}" data-book-id="${bookId}" ${canSaveToBookshelf ? '' : 'disabled'}>
                    <i class="fa-solid fa-bookmark"></i> ${canSaveToBookshelf ? (isInBookshelf ? '已加入書櫃' : '加入書櫃') : '無法收藏'}
                  </button>
                  <div class="total-score">${scoreBadgeLabel}: ${scoreBadgeValue}</div>
                </div>
        </div>
        
        <div class="tags">${tagsHtml}</div>
        <div class="intro">${item.intro || '暫無簡介...'}</div>
        
        ${breakdownHtml}
        ${explanationHtml}
    `;

  const toggleButton = card.querySelector('.bookshelf-toggle-btn');
  if (toggleButton && window.Bookshelf && canSaveToBookshelf) {
    toggleButton.addEventListener('click', function () {
      const saved = window.Bookshelf.toggleBook(item);
      setBookshelfButtonState(toggleButton, saved);
    });
  }

  return card;
}

function setBookshelfButtonState(button, saved) {
  if (!button) return;
  button.classList.toggle('saved', saved);
  button.innerHTML = `<i class="fa-solid fa-bookmark"></i> ${saved ? '已加入書櫃' : '加入書櫃'}`;
}

function syncBookshelfButtons() {
  if (!window.Bookshelf) return;
  const buttons = document.querySelectorAll('.bookshelf-toggle-btn[data-book-id]');
  buttons.forEach(button => {
    const bookId = button.getAttribute('data-book-id');
    const saved = bookId ? window.Bookshelf.hasBookId(bookId) : false;
    setBookshelfButtonState(button, saved);
  });
}

// 輔助函式：切換解釋區塊顯示
window.toggleExplanation = function (headerElement) {
  const content = headerElement.nextElementSibling;
  const icon = headerElement.querySelector('.fa-chevron-down') || headerElement.querySelector('.fa-chevron-up');

  if (content.classList.contains('hidden')) {
    content.classList.remove('hidden');
    if (icon) {
      icon.classList.remove('fa-chevron-down');
      icon.classList.add('fa-chevron-up');
    }
  } else {
    content.classList.add('hidden');
    if (icon) {
      icon.classList.remove('fa-chevron-up');
      icon.classList.add('fa-chevron-down');
    }
  }
}

// 輔助函式：不同評分類型給不同顏色
function getColorForCriteria(criteriaName) {
  switch (criteriaName) {
    case 'semantic_similarity': return '#4a90e2'; // 藍色 (向量語意)
    case 'numeric_range': return '#66bb6a'; // 綠色 (字數範圍)
    case 'keyword_match': return '#9c27b0'; // 紫色 (關鍵字/標籤)
    case 'status_check': return '#ff9800'; // 橘色 (狀態)
    case 'author_match': return '#00bcd4'; // 青色 (作者)
    default: return '#ab47bc'; // 預設紫色
  }
}

// 支援按下 Enter 鍵搜尋
document.getElementById('query-input').addEventListener('keypress', function (e) {
  if (e.key === 'Enter') {
    performSearch();
  }
});

document.addEventListener('bookshelf:changed', function () {
  syncBookshelfButtons();
});
