import { NextRequest, NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY || process.env.HF_TOKEN || '';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();

    const headers: Record<string, string> = {};
    if (INTERNAL_API_KEY) {
      headers['X-API-Key'] = INTERNAL_API_KEY;
      headers['Authorization'] = `Bearer ${INTERNAL_API_KEY}`;
    }

    const response = await fetch(`${FASTAPI_URL}/api/v1/compare`, {
      method: 'POST',
      body: formData,
      headers,
    });

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      return NextResponse.json(
        {
          success: false,
          error: `Backend returned HTTP ${response.status} (${response.statusText}). Please check your Hugging Face Space status and ensure it is Public or building is complete.`,
          raw_response: text.slice(0, 300),
        },
        { status: response.status || 502 }
      );
    }

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.detail || data.error || 'Ensemble comparison failed on backend.' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error proxying to FastAPI compare endpoint:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Could not connect to FastAPI backend service.',
      },
      { status: 502 }
    );
  }
}
