import { NextRequest, NextResponse } from 'next/server';

// Server-side only URL: Never exposed to client browser
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY || '';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();

    const headers: Record<string, string> = {};
    if (INTERNAL_API_KEY) {
      headers['X-API-Key'] = INTERNAL_API_KEY;
    }

    const response = await fetch(`${FASTAPI_URL}/api/v1/predict`, {
      method: 'POST',
      body: formData,
      headers,
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.detail || 'Inference failed on backend service.' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error proxying to FastAPI predict endpoint:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Could not connect to FastAPI backend service. Ensure it is running.',
      },
      { status: 502 }
    );
  }
}
