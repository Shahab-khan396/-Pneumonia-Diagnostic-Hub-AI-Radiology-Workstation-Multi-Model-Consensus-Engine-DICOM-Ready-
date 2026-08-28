import { NextResponse } from 'next/server';
import { Client } from '@gradio/client';

const SPACE_ID = process.env.FASTAPI_URL || process.env.NEXT_PUBLIC_HF_SPACE_URL || 'Shahabkhan396/pneumonia-hub';
const HF_TOKEN = process.env.HF_TOKEN || '';

export async function GET() {
  try {
    const client = await Client.connect(SPACE_ID, {
      hf_token: HF_TOKEN && HF_TOKEN.startsWith('hf_') ? (HF_TOKEN as `hf_${string}`) : undefined,
    });

    const result = await client.predict('/samples', []);
    const data: any = result.data;
    return NextResponse.json({ success: true, samples: data });
  } catch (error: any) {
    console.error('Error fetching samples via Gradio Client:', error);
    // Fallback sample catalog
    return NextResponse.json({
      success: true,
      samples: [
        {
          id: 'sample_normal',
          title: 'Normal Chest Radiograph',
          description: 'Clear lung fields, normal cardiothoracic ratio, sharp costophrenic angles.',
          truth: 'Normal',
          category: 'Normal / Healthy',
          filename: 'normal_clear_lungs.jpg',
        },
        {
          id: 'sample_bacterial',
          title: 'Bacterial Lobar Consolidation',
          description: 'Dense right middle/lower lobe airspace opacification with air bronchograms.',
          truth: 'Bacterial Pneumonia',
          category: 'Bacterial Pneumonia',
          filename: 'bacterial_lobar_pneumonia.jpg',
        },
        {
          id: 'sample_viral',
          title: 'Viral Interstitial Infiltrates',
          description: 'Diffuse bilateral peribronchial thickening and patchy reticular infiltrates.',
          truth: 'Viral Pneumonia',
          category: 'Viral Pneumonia',
          filename: 'viral_interstitial_pneumonia.jpg',
        },
      ],
    });
  }
}

