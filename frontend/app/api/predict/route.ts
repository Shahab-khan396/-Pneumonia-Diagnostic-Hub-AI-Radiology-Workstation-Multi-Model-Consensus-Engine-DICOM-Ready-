import { NextRequest, NextResponse } from 'next/server';
import { callGradioApi, uploadFileToGradio } from '../gradioClient';

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

    // Handle file upload if provided
    let filePayload = null;
    if (file && file.size > 0) {
      filePayload = await uploadFileToGradio(file);
    }

    const payload = [
      filePayload,
      sample_id,
      model_choice,
      explain,
      generate_report,
      patient_id,
      patient_age,
      patient_gender,
      clinical_history,
      referring_physician,
    ];

    const result = await callGradioApi('predict', payload);
    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Error in predict endpoint:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Inference failed on ZeroGPU backend service.',
      },
      { status: 502 }
    );
  }
}
