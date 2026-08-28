import { NextResponse } from 'next/server';

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';
const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY || process.env.HF_TOKEN || '';

export async function GET() {
  try {
    const headers: Record<string, string> = {};
    if (INTERNAL_API_KEY) {
      headers['X-API-Key'] = INTERNAL_API_KEY;
      headers['Authorization'] = `Bearer ${INTERNAL_API_KEY}`;
    }

    const response = await fetch(`${FASTAPI_URL}/api/v1/samples`, {
      method: 'GET',
      headers,
      next: { revalidate: 300 }, // cache for 5 minutes
    });

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      return NextResponse.json(
        { success: false, error: 'Could not fetch sample catalog from backend.' },
        { status: response.status || 502 }
      );
    }

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.detail || 'Could not fetch sample catalog.' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error fetching samples from FastAPI:', error);
    return NextResponse.json(
      { success: false, error: 'Could not connect to FastAPI backend.' },
      { status: 502 }
    );
  }
}
