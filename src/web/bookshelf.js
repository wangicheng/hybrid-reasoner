(function () {
  const STORAGE_KEY = 'bookshelf_v1';
  let books = [];

  function readStorage() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      books = Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      console.error('Failed to load bookshelf from localStorage:', error);
      books = [];
    }
  }

  function writeStorage() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(books));
  }

  function normalizeBook(item) {
    const bookId = String(item?.id ?? item?.book_id ?? item?.name ?? '').trim();
    return {
      id: bookId,
      title: item?.name || '未知書名',
      author: item?.author || '未知作者',
      cover: item?.cover || item?.cover_url || '',
      added_at: new Date().toISOString()
    };
  }

  function findIndexById(bookId) {
    return books.findIndex(book => String(book.id) === String(bookId));
  }

  function isInBookshelf(item) {
    const bookId = String(item?.id ?? item?.book_id ?? '').trim();
    if (!bookId) return false;
    return findIndexById(bookId) !== -1;
  }

  function hasBookId(bookId) {
    return findIndexById(bookId) !== -1;
  }

  function addBook(item) {
    const normalized = normalizeBook(item);
    if (!normalized.id) {
      return false;
    }

    if (findIndexById(normalized.id) !== -1) {
      return true;
    }

    books.unshift(normalized);
    writeStorage();
    renderBookshelf();
    emitChanged();
    return true;
  }

  function removeBook(bookId) {
    const index = findIndexById(bookId);
    if (index === -1) {
      return false;
    }

    books.splice(index, 1);
    writeStorage();
    renderBookshelf();
    emitChanged();
    return true;
  }

  function toggleBook(item) {
    const bookId = String(item?.id ?? item?.book_id ?? '').trim();
    if (!bookId) {
      return false;
    }

    if (hasBookId(bookId)) {
      removeBook(bookId);
      return false;
    }

    addBook(item);
    return true;
  }

  function emitChanged() {
    document.dispatchEvent(new CustomEvent('bookshelf:changed', {
      detail: {
        count: books.length,
        books: books.slice()
      }
    }));
  }

  function renderBookshelf() {
    const listEl = document.getElementById('bookshelf-list');
    const countEl = document.getElementById('bookshelf-count');
    if (!listEl || !countEl) return;

    countEl.textContent = String(books.length);

    if (books.length === 0) {
      listEl.innerHTML = '<div class="bookshelf-empty">還沒有加入任何書，先從搜尋結果收藏一本吧。</div>';
      return;
    }

    const html = books.map(book => `
      <div class="bookshelf-item" data-book-id="${book.id}">
        <div class="bookshelf-item-main">
          <div class="bookshelf-item-title">${book.title}</div>
          <div class="bookshelf-item-meta">${book.author}</div>
        </div>
        <button class="bookshelf-remove-btn" data-remove-book-id="${book.id}" type="button">
          移除
        </button>
      </div>
    `).join('');

    listEl.innerHTML = html;
  }

  function bindBookshelfEvents() {
    const listEl = document.getElementById('bookshelf-list');
    if (!listEl) return;

    listEl.addEventListener('click', function (event) {
      const target = event.target;
      if (!(target instanceof Element)) return;

      const removeId = target.getAttribute('data-remove-book-id');
      if (removeId) {
        removeBook(removeId);
      }
    });
  }

  function init() {
    readStorage();
    renderBookshelf();
    bindBookshelfEvents();
    emitChanged();
  }

  window.Bookshelf = {
    init,
    toggleBook,
    isInBookshelf,
    hasBookId,
    removeBook,
    renderBookshelf
  };

  document.addEventListener('DOMContentLoaded', function () {
    init();
  });
})();
