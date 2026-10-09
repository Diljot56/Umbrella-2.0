// Setup Marked.js with syntax highlighting
marked.setOptions({
    highlight: function(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    }
});

// DOM Elements
const queryInput = document.getElementById('queryInput');
const btnAsk = document.getElementById('btnAsk');
const btnSearch = document.getElementById('btnSearch');
const loadingIndicator = document.getElementById('loadingIndicator');
const loadingText = document.getElementById('loadingText');
const resultsArea = document.getElementById('resultsArea');

const answerCard = document.getElementById('answerCard');
const answerContent = document.getElementById('answerContent');
const fallbackBadge = document.getElementById('fallbackBadge');

const sourcesContainer = document.getElementById('sourcesContainer');
const sourcesList = document.getElementById('sourcesList');
const diagnosticsList = document.getElementById('diagnosticsList');

// Event Listeners for Chips
document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        queryInput.value = btn.innerText;
    });
});

// Event Listeners for Actions
btnAsk.addEventListener('click', () => handleAction('/ask'));
btnSearch.addEventListener('click', () => handleAction('/search'));
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleAction('/ask');
});

function getBadgeColor(type) {
    type = type.toLowerCase();
    if (type === 'document' || type === 'doc') return 'bg-emerald-900/50 text-emerald-400 border-emerald-800/50';
    if (type === 'ticket') return 'bg-amber-900/50 text-amber-400 border-amber-800/50';
    if (type === 'code') return 'bg-indigo-900/50 text-indigo-400 border-indigo-800/50';
    return 'bg-slate-800 text-slate-300 border-slate-700';
}

function processMarkdownCitations(text) {
    return text.replace(/\[(\d+)\]/g, (match, id) => {
        return `<span class="citation-link" onmouseover="highlightSource('${id}')" onmouseout="unhighlightSource('${id}')" onclick="scrollToSource('${id}')">[${id}]</span>`;
    });
}

window.highlightSource = (id) => {
    const el = document.getElementById(`source-card-${id}`);
    if (el) el.classList.add('highlighted');
}
window.unhighlightSource = (id) => {
    const el = document.getElementById(`source-card-${id}`);
    if (el) el.classList.remove('highlighted');
}
window.scrollToSource = (id) => {
    const el = document.getElementById(`source-card-${id}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

window.togglePreview = (id) => {
    const el = document.getElementById(`preview-${id}`);
    if (el.classList.contains('line-clamp-3')) {
        el.classList.remove('line-clamp-3');
    } else {
        el.classList.add('line-clamp-3');
    }
}

async function handleAction(endpoint) {
    let query = queryInput.value.trim();
    if (!query) return;

    // Reset UI
    resultsArea.classList.add('hidden');
    answerCard.classList.add('hidden');
    sourcesContainer.classList.add('hidden');
    fallbackBadge.classList.add('hidden');
    sourcesList.innerHTML = '';
    diagnosticsList.innerHTML = '';
    
    loadingIndicator.classList.remove('hidden');
    loadingText.innerText = endpoint === '/ask' ? 'Synthesizing Answer...' : 'Retrieving & Reranking...';
    loadingText.className = `text-lg font-medium animate-pulse ${endpoint === '/ask' ? 'text-brand-400' : 'text-slate-400'}`;

    try {
        const payload = endpoint === '/ask' ? {
            query: query, top_dense: 15, top_sparse: 15, top_final: 5
        } : {
            query: query, top_dense: 15, top_sparse: 15, rrf_k: 60, top_final: 10
        };

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        const data = await response.json();

        renderResults(endpoint, data);

    } catch (err) {
        console.error(err);
        alert('An error occurred. Make sure the backend is running.');
    } finally {
        loadingIndicator.classList.add('hidden');
        resultsArea.classList.remove('hidden');
    }
}

function renderResults(endpoint, data) {
    let retrieved_chunks = endpoint === '/ask' ? data.retrieved_chunks : data;

    if (endpoint === '/ask') {
        answerCard.classList.remove('hidden');
        sourcesContainer.classList.remove('hidden');
        
        if (data.answer.includes("No supported answer could be found in the provided sources.")) {
            fallbackBadge.classList.remove('hidden');
        }

        answerContent.innerHTML = marked.parse(processMarkdownCitations(data.answer));
        
        // Render Citations
        data.citations.forEach(cit => {
            const num = cit.marker.replace(/\[|\]/g, '');
            const badgeClass = getBadgeColor(cit.source_type);
            
            const card = document.createElement('div');
            card.id = `source-card-${num}`;
            card.className = "source-card glass-panel rounded-xl p-5 border border-slate-700/50 relative overflow-hidden group flex flex-col gap-3";
            card.innerHTML = `
                <div class="flex justify-between items-start">
                    <div class="flex items-center gap-3">
                        <span class="bg-brand-900/60 text-brand-400 rounded-full w-8 h-8 flex items-center justify-center font-bold font-mono border border-brand-700/50 text-sm">[${num}]</span>
                        <div class="flex flex-col">
                            <span class="font-semibold text-slate-200 text-sm truncate max-w-[250px] md:max-w-md" title="${cit.source_id}">${cit.source_id}</span>
                            <span class="text-xs text-slate-500 font-mono">${cit.location}</span>
                        </div>
                    </div>
                    <span class="text-[10px] font-bold px-2 py-0.5 rounded border ${badgeClass} tracking-widest uppercase">${cit.source_type}</span>
                </div>
                <div class="bg-slate-900/60 p-3 rounded-lg border border-slate-700/30">
                    <p id="preview-${num}" class="text-sm text-slate-400 font-mono leading-relaxed line-clamp-3 cursor-pointer" onclick="togglePreview('${num}')">${cit.snippet}</p>
                </div>
            `;
            sourcesList.appendChild(card);
        });
    } else {
        answerCard.classList.add('hidden');
        sourcesContainer.classList.add('hidden');
        resultsArea.classList.remove('lg:grid-cols-3');
        resultsArea.classList.add('lg:grid-cols-1');
    }

    if (endpoint === '/search') {
        resultsArea.className = 'grid grid-cols-1 gap-6'; 
    } else {
        resultsArea.className = 'grid grid-cols-1 lg:grid-cols-3 gap-6';
    }

    // Render Diagnostics
    retrieved_chunks.forEach((chunk, i) => {
        let sc = chunk.scores;
        let isDense = sc.dense_rank !== null;
        let isSparse = sc.bm25_rank !== null;
        
        let insightBadge = isDense && isSparse ? '<span class="bg-brand-900/40 text-brand-400 border border-brand-800 text-[10px] px-1.5 py-0.5 rounded ml-2">Hybrid Consensus</span>' : '';
        if(isDense && !isSparse) insightBadge = '<span class="bg-purple-900/40 text-purple-400 border border-purple-800 text-[10px] px-1.5 py-0.5 rounded ml-2">Dense Match</span>';
        if(!isDense && isSparse) insightBadge = '<span class="bg-orange-900/40 text-orange-400 border border-orange-800 text-[10px] px-1.5 py-0.5 rounded ml-2">Exact Overlap</span>';

        const formatFixed = (val) => val !== null && val !== undefined ? Number(val).toFixed(4) : 'N/A';

        const row = document.createElement('div');
        row.className = "bg-slate-800/40 rounded-xl p-4 border border-slate-700/50 hover:bg-slate-800/60 transition-colors flex flex-col gap-3";
        row.innerHTML = `
            <div class="flex justify-between items-start">
                <div class="flex items-center gap-2">
                    <span class="text-slate-500 font-mono text-xs">#${i+1}</span>
                    <span class="text-sm font-medium text-slate-300 truncate max-w-[200px]" title="${chunk.source_id} - ${chunk.location}">${chunk.source_id}</span>
                </div>
                <div class="flex items-center">
                    <span class="text-[10px] font-bold px-1.5 py-0.5 rounded border ${getBadgeColor(chunk.source_type)}">${chunk.source_type.substring(0,3).toUpperCase()}</span>
                </div>
            </div>
            
            <p class="text-xs text-slate-500 font-mono bg-slate-900/50 p-2 rounded truncate overflow-hidden">${chunk.text.substring(0,60)}...</p>

            <div class="grid grid-cols-2 gap-2 text-xs mt-1">
                <div class="flex justify-between items-center bg-slate-900/40 p-1.5 rounded">
                    <span class="text-slate-500">CE Final:</span>
                    <span class="font-mono text-brand-400 font-bold">${formatFixed(sc.cross_encoder_score)}</span>
                </div>
                <div class="flex justify-between items-center bg-slate-900/40 p-1.5 rounded">
                    <span class="text-slate-500">RRF:</span>
                    <span class="font-mono text-amber-400">${formatFixed(sc.rrf_score)}</span>
                </div>
                <div class="flex flex-col bg-slate-900/40 p-1.5 rounded">
                    <div class="flex justify-between"><span class="text-slate-500">Dense Score:</span><span class="font-mono text-sky-300">${formatFixed(sc.dense_score)}</span></div>
                    <div class="flex justify-between"><span class="text-slate-500">Rank:</span><span class="font-mono">${sc.dense_rank || '-'}</span></div>
                </div>
                <div class="flex flex-col bg-slate-900/40 p-1.5 rounded">
                    <div class="flex justify-between"><span class="text-slate-500">BM25 Score:</span><span class="font-mono text-emerald-300">${formatFixed(sc.bm25_score)}</span></div>
                    <div class="flex justify-between"><span class="text-slate-500">Rank:</span><span class="font-mono">${sc.bm25_rank || '-'}</span></div>
                </div>
            </div>
            <div class="flex justify-start mt-1">
                ${insightBadge}
            </div>
        `;
        diagnosticsList.appendChild(row);
    });
}
