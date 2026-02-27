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
      displayInterpretation(interpretationBox, data.parsed_criteria, data.query_vector);
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

function displayInterpretation(container, criteriaList, queryVector) {
  container.classList.remove('hidden');

  // Group by type for cleaner display
  const tags = [];
  const keywords = [];
  let semanticQuery = null;

  criteriaList.forEach(c => {
    if (c.name === 'keyword_match') {
      const field = c.parameters.field;
      const kw = c.parameters.keyword;
      if (field === 'tags') tags.push(kw);
      else keywords.push(`${field}:${kw}`);
    } else if (c.name === 'semantic_similarity') {
      semanticQuery = c.parameters.query_text;
    }
  });

  let html = `<h3><i class="fa-solid fa-robot"></i> AI 搜尋解析</h3>`;
  html += `<div>`;

  if (tags.length > 0) {
    html += `<strong><i class="fa-solid fa-tags"></i> 鎖定標籤：</strong> `;
    html += tags.map(t => `<span class="criteria-tag">${t}</span>`).join('');
    html += `<br>`;
  }

  if (keywords.length > 0) {
    html += `<strong><i class="fa-solid fa-filter"></i> 關鍵字篩選：</strong> `;
    html += keywords.map(k => `<span class="criteria-tag">${k}</span>`).join('');
    html += `<br>`;
  }

  if (semanticQuery) {
    html += `<div style="margin-top:5px; font-size:0.9em; color:#555;">`;
    html += `<strong><i class="fa-solid fa-layer-group"></i> 語意檢索：</strong> 使用向量搜尋相近內容`;
    // Only show if it's different from tags (to avoid clutter)
    if (tags.length === 0) {
      html += ` ("${semanticQuery}")`;
    }
    html += `</div>`;
  }

  if (queryVector && queryVector.length > 0) {
    const dim = queryVector.length;
    const preview = queryVector.slice(0, 5).map(v => v.toFixed(3)).join(', ');
    html += `<div style="margin-top:8px; padding-top:8px; border-top:1px dashed #ccc; font-size:0.85em; color:#666;">`;
    html += `<strong><i class="fa-solid fa-location-crosshairs"></i> 向量座標：</strong> [${preview}, ...] <span style="background:#eee; padding:2px 6px; border-radius:4px; font-size:0.8em;">${dim} 維度</span>`;
    html += `</div>`;
  }

  html += `</div>`;

  // 加入顯示原始 Payload 的按鈕和區塊
  const payloadJson = JSON.stringify(criteriaList, null, 2);
  html += `
    <div style="margin-top: 15px; border-top: 1px dashed #ccc; padding-top: 10px;">
      <div onclick="togglePayload(this)" style="cursor: pointer; color: var(--primary-color); font-size: 0.9em; display: inline-block;">
        <i class="fa-solid fa-code"></i> <span class="toggle-text">顯示查詢 Payload JSON</span>
      </div>
      <pre class="raw-payload hidden" style="background: #282c34; color: #abb2bf; padding: 12px; border-radius: 6px; font-size: 0.85em; overflow-x: auto; margin-top: 10px; max-height: 300px;">${payloadJson}</pre>
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

  // 1. 基本資訊與標籤
  const tagsHtml = (item.tags || []).map(tag => `<span class="tag">#${tag}</span>`).join('');

  // 2. 評分視覺化 (Breakdown Visualization)
  // 我們將每個評分項目的 weighted_score 繪製成條狀圖
  // 為了顯示比例，我們假設滿分是 1.0 (或根據實際總分動態調整)
  let breakdownHtml = '<div class="score-breakdown">';
  if (result.breakdown) {
    result.breakdown.forEach(b => {
      const originalScore = b.normalized_score !== undefined ? b.normalized_score : b.raw_score;

      // 以單項 0~1 的 originalScore 作為百分比顯示
      const widthPercent = Math.min(originalScore * 100, 100);
      const label = b.label || b.criteria;

      breakdownHtml += `
                <div class="score-bar-container">
                    <span class="score-label">${label}</span>
                    <span style="font-size:0.8em; color:#666;">${originalScore.toFixed(3)}</span>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${widthPercent}%; background-color: ${getColorForCriteria(b.criteria)}"></div>
                    </div>
                </div>
            `;
    });
  }
  breakdownHtml += '</div>';

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
            <div class="total-score">Score: ${result.score.toFixed(4)}</div>
        </div>
        
        <div class="tags">${tagsHtml}</div>
        <div class="intro">${item.intro || '暫無簡介...'}</div>
        
        ${breakdownHtml}
        ${explanationHtml}
    `;

  return card;
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
