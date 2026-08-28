import { NextResponse } from 'next/server';
import { callGradioApi } from '../gradioClient';

export async function GET() {
  try {
    const result = await callGradioApi('samples', []);
    return NextResponse.json({ success: true, samples: result });
  } catch (error: any) {
    console.error('Error fetching samples from Gradio API:', error);
    // Graceful fallback to default samples catalog
    return NextResponse.json({
      success: true,
      samples: [
        {
          id: 'sample_normal',
          filename: 'normal_clear_lungs.jpg',
          title: 'Normal Radiograph',
          category: 'NORMAL',
          subtitle: 'Clear bilateral lung fields, sharp costophrenic angles',
          description: 'Healthy adult radiograph displaying normal bronchovascular arborization without focal consolidation.',
          badge: 'Normal CXR',
          badge_class: 'badge-normal',
          image_url: '/static/samples/normal_clear_lungs.jpg',
        },
        {
          id: 'sample_bacterial',
          filename: 'bacterial_lobar_pneumonia.jpg',
          title: 'Bacterial Pneumonia',
          category: 'PNEUMONIA',
          subtitle: 'Dense right middle lobe alveolar consolidation',
          description: 'Demonstrates classical lobar consolidation with air bronchograms typical of bacterial Streptococcus infection.',
          badge: 'Bacterial Lobar',
          badge_class: 'badge-pneumonia',
          image_url: '/static/samples/bacterial_lobar_pneumonia.jpg',
        },
        {
          id: 'sample_viral',
          filename: 'viral_interstitial_pneumonia.jpg',
          title: 'Viral Pneumonia',
          category: 'PNEUMONIA',
          subtitle: 'Bilateral diffuse interstitial & reticular opacities',
          description: 'Shows diffuse peribronchial thickening and ground-glass haziness typical of viral etiology.',
          badge: 'Viral Interstitial',
          badge_class: 'badge-pneumonia',
          image_url: '/static/samples/viral_interstitial_pneumonia.jpg',
        },
      ],
    });
  }
}
