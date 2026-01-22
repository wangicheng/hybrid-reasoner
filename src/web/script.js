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
    
    // --- Display Search Interpretation ---
    if (data.parsed_criteria && data.parsed_criteria.length > 0) {
        displayInterpretation(interpretationBox, data.parsed_criteria, data.query_vector);
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
    resultsContainer.innerHTML = '<div style="color:red; text-align:center;">發生錯誤，請檢查後端服務是否啟動</div>';
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
    container.innerHTML = html;
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
      // 計算寬度百分比 (簡單起見，將 weighted_score 放大顯示，或相對於 total score)
      // 這裡假設單項滿分貢獻約為 0.5~1.0，我們用一個視覺係數放大它以便觀察
      const widthPercent = Math.min((b.weighted_score / result.score) * 100, 100);
      const label = getCriteriaLabel(b);

      breakdownHtml += `
                <div class="score-bar-container">
                    <span class="score-label">${label}</span>
                    <span style="font-size:0.8em; color:#666;">${b.weighted_score.toFixed(3)}</span>
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
    explanationHtml = `
            <div class="ai-explanation-box">
                <div class="ai-header" onclick="toggleExplanation(this)">
                    <span><i class="fa-solid fa-robot"></i> AI 推薦理由</span>
                    <i class="fa-solid fa-chevron-down"></i>
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
                    <i class="fa-solid fa-pen-nib"></i> ${item.word_count ? item.word_count.toLocaleString() : '未知'}字
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

// 輔助函式：根據 breakdown 項目生成人類可讀的標籤
function getCriteriaLabel(breakdownItem) {
  const name = breakdownItem.criteria;
  const params = breakdownItem.params || {};

  switch (name) {
    case 'semantic_similarity':
      return '語意與內容相似度';
    case 'keyword_match':
      // 顯示具體匹配到的關鍵字
      const keyword = params.keyword || '關鍵字';
      const field = params.field === 'classification' ? '分類' : (params.field === 'tags' ? '標籤' : params.field || '');
      return `${field}: ${keyword}`;
    case 'numeric_range':
      // 顯示具體的數值範圍
      const min = params.min_val;
      const max = params.max_val;
      let rangeStr = '字數範圍';
      if (min && max) {
        rangeStr = `字數: ${(min / 10000).toFixed(0)}萬 - ${(max / 10000).toFixed(0)}萬`;
      } else if (min) {
        rangeStr = `字數 > ${(min / 10000).toFixed(0)}萬`;
      } else if (max) {
        rangeStr = `字數 < ${(max / 10000).toFixed(0)}萬`;
      }
      return rangeStr;
    case 'status_check':
      const status = params.target_status || '狀態';
      return `狀態: ${status}`;
    case 'author_match':
      const author = params.author_name || '作者';
      return `作者: ${author}`;
    default:
      return name;
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
