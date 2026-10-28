const fileInput = document.querySelector('#audioFile');
const dropZone = document.querySelector('#dropZone');
const analyzeButton = document.querySelector('#analyzeButton');
const recordButton = document.querySelector('#recordButton');
const clearButton = document.querySelector('#clearButton');
const emptyState = document.querySelector('#emptyState');
const resultContent = document.querySelector('#resultContent');
const fileName = document.querySelector('#fileName');
const fileMeta = document.querySelector('#fileMeta');
const resultTitle = document.querySelector('#resultTitle');
const classification = document.querySelector('#classification');
const verdictIcon = document.querySelector('#verdictIcon');
const verdictLabel = document.querySelector('#verdictLabel');
const verdictCopy = document.querySelector('#verdictCopy');
const scoreRow = document.querySelector('.score-row');
const scoreMeter = document.querySelector('#scoreMeter');

let selectedFile;
function formatSize(bytes) { return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`; }
function selectFile(file) {
  if (!file || !file.type.startsWith('audio/')) { alert('Please choose a supported audio file.'); return; }
  selectedFile = file; fileName.textContent = file.name; fileMeta.textContent = `${formatSize(file.size)} · Ready for analysis`;
  analyzeButton.disabled = false; analyzeButton.textContent = 'Analyze voice';
}
fileInput.addEventListener('change', e => selectFile(e.target.files[0]));
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
['dragenter','dragover'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave','drop'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', e => selectFile(e.dataTransfer.files[0]));
analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeButton.textContent = 'Analyzing…'; analyzeButton.disabled = true; resultTitle.textContent = 'Scanning voiceprint';
  try {
    const data = new FormData();
    data.append('audio', selectedFile);
    const response = await fetch('/analyze', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Analysis failed.');

    const isDeepfake = result.label === 'Deepfake Voice';
    emptyState.classList.add('hidden'); resultContent.classList.remove('hidden');
    resultTitle.textContent = 'Analysis complete';
    verdictLabel.textContent = result.label.toUpperCase();
    verdictCopy.textContent = isDeepfake ? 'Possible AI / synthetic voice detected.' : 'Voice classified as genuine human.';
    verdictIcon.textContent = isDeepfake ? '!' : '✓';
    document.querySelector('#scoreText').textContent = `${result.confidence}%`;
    scoreMeter.style.width = `${result.confidence}%`;
    document.querySelector('#cnnScore').textContent = result.cnnScore.toFixed(6);
    document.querySelector('#riskLevel').textContent = result.risk;
    document.querySelector('#deepfakeLikelihood').textContent = `${result.deepfakeLikelihood}%`;
    classification.classList.toggle('is-deepfake', isDeepfake);
    scoreRow.classList.toggle('is-deepfake', isDeepfake);
    scoreMeter.parentElement.classList.toggle('is-deepfake', isDeepfake);
  } catch (error) {
    resultTitle.textContent = 'Analysis unavailable';
    alert(error.message);
  } finally {
    analyzeButton.textContent = 'Analyze voice';
    analyzeButton.disabled = false;
  }
});
clearButton.addEventListener('click', () => { selectedFile = null; fileInput.value = ''; resultContent.classList.add('hidden'); emptyState.classList.remove('hidden'); resultTitle.textContent = 'Ready to analyze'; analyzeButton.textContent = 'Analyze voice'; analyzeButton.disabled = true; });
recordButton.addEventListener('click', () => { alert('Recording can be connected to your browser microphone or backend service in the next step.'); });
