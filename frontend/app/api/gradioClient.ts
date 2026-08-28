/**
 * Robust ZeroGPU Gradio 5 API Client for Pneumonia Diagnostic Hub
 * Directly communicates with Hugging Face ZeroGPU Gradio endpoints without SDK mismatch or CORS issues.
 */

const RAW_SPACE_URL = process.env.FASTAPI_URL || process.env.NEXT_PUBLIC_HF_SPACE_URL || 'https://shahabkhan396-pneumonia-hub.hf.space';
const HF_TOKEN = process.env.HF_TOKEN || '';

export function getBaseSpaceUrl(): string {
  let url = RAW_SPACE_URL.trim();
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    url = `https://${url}`;
  }
  if (!url.includes('.hf.space') && url.includes('huggingface.co/spaces/')) {
    const parts = url.split('spaces/')[1]?.split('/');
    if (parts && parts.length >= 2) {
      url = `https://${parts[0]}-${parts[1]}.hf.space`;
    }
  }
  return url.replace(/\/$/, '');
}

/**
 * Uploads a file to Hugging Face Gradio Space and returns the file payload
 */
export async function uploadFileToGradio(file: File): Promise<{ path: string; meta: { _type: string } }> {
  const baseUrl = getBaseSpaceUrl();
  const formData = new FormData();
  formData.append('files', file, file.name);

  const headers: Record<string, string> = {};
  if (HF_TOKEN) {
    headers['Authorization'] = `Bearer ${HF_TOKEN}`;
  }

  const uploadRes = await fetch(`${baseUrl}/gradio_api/upload`, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!uploadRes.ok) {
    throw new Error(`Failed to upload scan to Hugging Face Space (HTTP ${uploadRes.status})`);
  }

  const uploaded = await uploadRes.json();
  const filePath = Array.isArray(uploaded) ? uploaded[0] : uploaded?.path || uploaded;

  return {
    path: filePath,
    meta: { _type: 'gradio.FileData' },
  };
}

/**
 * Calls a Gradio 5 API endpoint with ZeroGPU queue support and returns the parsed JSON response.
 */
export async function callGradioApi(endpointName: string, dataPayload: any[]): Promise<any> {
  const baseUrl = getBaseSpaceUrl();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (HF_TOKEN) {
    headers['Authorization'] = `Bearer ${HF_TOKEN}`;
  }

  // 1. Submit job to Gradio 5 ZeroGPU queue
  const initRes = await fetch(`${baseUrl}/gradio_api/call/${endpointName}`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ data: dataPayload }),
  });

  if (!initRes.ok) {
    const errText = await initRes.text().catch(() => '');
    throw new Error(`Gradio API initiation failed (HTTP ${initRes.status}): ${errText.slice(0, 200)}`);
  }

  const { event_id } = await initRes.json();
  if (!event_id) {
    throw new Error('Gradio API did not return a valid queue event ID.');
  }

  // 2. Fetch event result stream
  const eventRes = await fetch(`${baseUrl}/gradio_api/call/${endpointName}/${event_id}`, {
    headers,
  });

  if (!eventRes.ok) {
    throw new Error(`Failed to retrieve inference event result (HTTP ${eventRes.status})`);
  }

  const rawText = await eventRes.text();
  for (const line of rawText.split('\n')) {
    if (line.startsWith('data: ')) {
      const parsed = JSON.parse(line.slice(6));
      return Array.isArray(parsed) ? parsed[0] : parsed;
    }
  }

  throw new Error('No completion data received from ZeroGPU stream.');
}
