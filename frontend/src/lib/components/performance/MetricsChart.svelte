<script lang="ts">
  import { onMount } from 'svelte';
  import type { PerformanceMetric } from '$lib/performance/metrics-collector';

  interface Props {
    metrics: PerformanceMetric[];
    metric: string;
    window: number;
  }

  let { metrics, metric, window }: Props = $props();
  
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  
  let filteredMetrics = $derived(() => {
    const cutoff = Date.now() - window;
    return metrics
      .filter(m => m.name === metric && m.timestamp >= cutoff)
      .sort((a, b) => a.timestamp - b.timestamp);
  });

  onMount(() => {
    ctx = canvas.getContext('2d');
    if (ctx) {
      drawChart();
    }
  });

  $effect(() => {
    if (ctx && filteredMetrics) {
      drawChart();
    }
  });

  function drawChart() {
    if (!ctx || !canvas) return;

    const data = filteredMetrics();
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (data.length === 0) {
      // Draw "No data" message
      ctx.fillStyle = '#6b7280';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data available', canvas.width / 2, canvas.height / 2);
      return;
    }

    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Calculate scales
    const minTime = Math.min(...data.map(d => d.timestamp));
    const maxTime = Math.max(...data.map(d => d.timestamp));
    const minValue = Math.min(...data.map(d => d.value));
    const maxValue = Math.max(...data.map(d => d.value));
    const valueRange = maxValue - minValue || 1;

    const xScale = (timestamp: number) => 
      padding.left + ((timestamp - minTime) / (maxTime - minTime || 1)) * chartWidth;
    
    const yScale = (value: number) => 
      padding.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;

    // Draw axes
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#f3f4f6';
    ctx.lineWidth = 0.5;
    
    // Horizontal grid lines
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const y = padding.top + (chartHeight / ySteps) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
      
      // Y-axis labels
      const value = maxValue - (valueRange / ySteps) * i;
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(formatValue(value), padding.left - 10, y + 4);
    }

    // Draw line chart
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((point, index) => {
      const x = xScale(point.timestamp);
      const y = yScale(point.value);
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();

    // Draw points
    ctx.fillStyle = '#3b82f6';
    data.forEach(point => {
      const x = xScale(point.timestamp);
      const y = yScale(point.value);
      
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw time labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    const xSteps = Math.min(5, data.length);
    for (let i = 0; i <= xSteps; i++) {
      const index = Math.floor((data.length - 1) * (i / xSteps));
      const point = data[index];
      if (point) {
        const x = xScale(point.timestamp);
        const time = new Date(point.timestamp).toLocaleTimeString();
        ctx.fillText(time, x, padding.top + chartHeight + 20);
      }
    }

    // Draw title
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(getMetricLabel(metric), canvas.width / 2, 15);
  }

  function formatValue(value: number): string {
    if (metric === 'memory_used') {
      return `${(value / 1024 / 1024).toFixed(0)}MB`;
    }
    if (metric === 'frame_rate') {
      return `${value.toFixed(0)}fps`;
    }
    if (metric === 'cache_hit_rate') {
      return `${(value * 100).toFixed(0)}%`;
    }
    return `${value.toFixed(0)}ms`;
  }

  function getMetricLabel(metric: string): string {
    const labels: Record<string, string> = {
      response_time: 'Response Time',
      memory_used: 'Memory Usage',
      frame_rate: 'Frame Rate',
      cache_hit_rate: 'Cache Hit Rate',
      long_task: 'Long Tasks'
    };
    return labels[metric] || metric;
  }
</script>

<div class="metrics-chart">
  <canvas 
    bind:this={canvas}
    width={800}
    height={300}
  ></canvas>
</div>

<style>
  .metrics-chart {
    width: 100%;
    overflow-x: auto;
  }

  canvas {
    width: 100%;
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border-color, #e5e7eb);
    border-radius: 0.375rem;
    background: white;
  }
</style>