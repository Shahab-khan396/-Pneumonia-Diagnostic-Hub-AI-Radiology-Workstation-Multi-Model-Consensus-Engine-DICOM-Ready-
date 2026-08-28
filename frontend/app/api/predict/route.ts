import { NextRequest, NextResponse } from 'next/server';
import { Client } from '@gradio/client';

const SPACE_ID = process.env.FASTAPI_URL || process.env.NEXT_PUBLIC_HF_SPACE_URL || 'Shahabkhan396/pneumonia-hub';
const HF_TOKEN = process.env.HF_TOKEN || '';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;
    const sample_id = (formData.get('sample_id') as string) || '';
    const model_choice = (formData.get('model_choice') as string) || 'mobilenet';
    const explain = formData.get('explain') === 'true';
    const generate_report = formData.get('generate_report') === 'true';
    const patient_id = (formData.get('patient_id') as string) || '';
    const patient_age = (formData.get('patient_age') as string) || '';
    const patient_gender = (formData.get('patient_gender') as string) || '';
    const clinical_history = (formData.get('clinical_history') as string) || '';
    const referring_physician = (formData.get('referring_physician') as string) || '';

    // Connect to Gradio Space Client
    const target = SPACE_ID.includes('/') && !SPACE_ID.startsWith('http')
      ? SPACE_ID
      : (SPACE_ID.replace('https://', '').replace('.hf.space', '').replace('/', '/'));

    const client = await Client.connect(SPACE_ID, {
      hf_token: HF_TOKEN && HF_TOKEN.startsWith('hf_') ? (HF_TOKEN as `hf_${string}`) : undefined,
    });

    const fileBlob = file ? file : null;

    const result = await client.predict('/predict', [
      fileBlob,
      sample_id,
      model_choice,
      explain,
      generate_report,
      patient_id,
      patient_age,
      patient_gender,
      clinical_history,
      referring_physician,
    ]);

    const data: any = result.data;
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error proxying to Gradio predict endpoint:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Could not connect to AI backend service. Ensure Hugging Face Space is running.',
      },
      { status: 502 }
    );
  }
}

