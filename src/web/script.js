document.addEventListener('DOMContentLoaded', () => {
  const searchBtn = document.getElementById('searchBtn');
  const queryInput = document.getElementById('queryInput');
  const loading = document.getElementById('loading');
  const resultsArea = document.getElementById('resultsArea');
  const criteriaSection = document.getElementById('criteriaSection');
  const criteriaTags = document.getElementById('criteriaTags');
  const resultsList = document.getElementById('resultsList');

  // Allow Enter key to search
  queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
  });

  searchBtn.addEventListener('click', performSearch);

  async function performSearch() {
    const query = queryInput.value.trim();
    if (!query) return;

    // UI Reset
    loading.classList.remove('hidden');
    resultsArea.classList.add('hidden');
    resultsList.innerHTML = '';
    criteriaTags.innerHTML = '';

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) throw new Error("Search failed");

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error(err);
      alert("搜尋發生錯誤，請稍後再試。");
    } finally {
      loading.classList.add('hidden');
      resultsArea.classList.remove('hidden');
    }
  }

  function renderResults(data) {
    // Render Criteria
    if (data.parsed_criteria && data.parsed_criteria.length > 0) {
      criteriaSection.classList.remove('hidden');
      criteriaTags.innerHTML = ''; // Clear previous
      data.parsed_criteria.forEach(c => {
        const item = document.createElement('div');
        item.className = 'criteria-item';

        // Format parameters nicely
        // specific definitions for what to show per criteria type
        const validParamsMap = {
          'keyword_match': ['field', 'keyword'],
          'numeric_range': ['field', 'min_val', 'max_val'],
          'status_check': ['target_status'],
          'author_match': ['author_name'],
          'is_free_check': ['require_free'],
          'age_check': ['allow_restricted'],
          'audio_available': ['require_audio'],
          'semantic_similarity': ['query_text']
        };

        let paramsHtml = '';
        if (c.parameters) {
          const allowedKeys = validParamsMap[c.name] || Object.keys(c.parameters);

          const entries = Object.entries(c.parameters).filter(([k, v]) => {
            return v !== null && v !== undefined && allowedKeys.includes(k);
          });

          if (entries.length > 0) {
            paramsHtml = '<div class="criteria-params">';
            entries.forEach(([k, v]) => {
              paramsHtml += `<span class="param-pill">${k}: ${v}</span>`;
            });
            paramsHtml += '</div>';
          }
        }

        item.innerHTML = `
            <div class="criteria-header">
                <span class="criteria-name"><i class="fa-solid fa-filter"></i> ${c.name}</span>
                <span class="criteria-weight">Weight: ${c.weight}</span>
            </div>
            <div class="criteria-desc">${c.description || ''}</div>
            ${paramsHtml}
        `;
        criteriaTags.appendChild(item);
      });
    } else {
      criteriaSection.classList.add('hidden');
    }

    // Render Cards
    if (!data.results || data.results.length === 0) {
      resultsList.innerHTML = '<p style="text-align:center; color:#9ca3af;">找不到相關結果。</p>';
      return;
    }

    data.results.forEach(res => {
      const item = res.item;
      const score = res.score.toFixed(4);
      const breakdown = res.breakdown;

      const card = document.createElement('div');
      card.className = 'result-card';

      // Generate breakdown HTML
      let breakdownHtml = '<div class="breakdown-content">';
      breakdownHtml += '<h4>得分詳情</h4>';
      breakdown.forEach(b => {
        const wScore = b.weighted_score !== undefined ? b.weighted_score.toFixed(4) : 'Error';
        const rScore = b.raw_score !== undefined ? b.raw_score.toFixed(4) : 'N/A';
        breakdownHtml += `
                    <div class="score-row">
                        <span class="score-label">${b.criteria} (W: ${b.weight}):</span>
                        <span class="score-val">Raw: ${rScore} -> +${wScore}</span>
                    </div>`;
      });
      breakdownHtml += '</div>';

      // Tags
      let tagsHtml = '';
      if (item.tags && Array.isArray(item.tags)) {
        item.tags.slice(0, 5).forEach(t => {
          tagsHtml += `<span class="small-tag">#${t}</span>`;
        });
      }

      const statusIcon = item.publish_status === 'completed' ? '<i class="fa-solid fa-check-circle"></i>' : '<i class="fa-solid fa-pen-nib"></i>';
      const statusText = item.publish_status === 'completed' ? '完結' : '連載中';

      // AI Explanation HTML
      let explanationHtml = '';
      if (res.explanation) {
        explanationHtml = `
          <div class="ai-explanation">
            <div class="ai-explanation-header">
              <i class="fa-solid fa-wand-magic-sparkles"></i> AI 推薦理由
            </div>
            <p class="ai-explanation-text">${res.explanation}</p>
          </div>
        `;
      }

      card.innerHTML = `
                <div class="card-header">
                    <div>
                        <h2 class="novel-title">${item.name || '未命名'}</h2>
                        <div class="novel-meta">
                            <span class="meta-item"><i class="fa-solid fa-user"></i> ${item.author || '未知作者'}</span>
                            <span class="meta-item">${statusIcon} ${statusText}</span>
                            <span class="meta-item"><i class="fa-solid fa-layer-group"></i> ${item.classification || '一般'}</span>
                        </div>
                    </div>
                    <div class="novel-score">${score}</div>
                </div>
                
                <div class="tags-container">${tagsHtml}</div>
                <p class="novel-intro">${item.intro || '無簡介...'}</p>
                
                ${explanationHtml}
                
                <button class="breakdown-toggle">顯示推理過程 <i class="fa-solid fa-chevron-down"></i></button>
                ${breakdownHtml}
            `;

      // Toggle Logic
      const toggleBtn = card.querySelector('.breakdown-toggle');
      const content = card.querySelector('.breakdown-content');
      toggleBtn.addEventListener('click', () => {
        const isHidden = getComputedStyle(content).display === 'none';
        content.style.display = isHidden ? 'block' : 'none';
        toggleBtn.querySelector('i').classList.toggle('fa-chevron-down');
        toggleBtn.querySelector('i').classList.toggle('fa-chevron-up');
      });

      resultsList.appendChild(card);
    });
  }
});
