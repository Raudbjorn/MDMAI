<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { ProviderType } from '$lib/types/providers';
	import type { AIProviderStats } from '$lib/types/providers';
	
	// Store data
	let stats = $derived(providerStore.stats);
	let health = $derived(providerStore.health);
	
	// Local state
	let selectedMetric = $state<'requests' | 'tokens' | 'cost' | 'latency'>('requests');
	let selectedTimeRange = $state<'24h' | '7d' | '30d'>('7d');
	let showDetails = $state(false);
	
	// Provider names and colors
	const providerConfig: Record<ProviderType, { name: string; color: string }> = {
		[ProviderType.ANTHROPIC]: { name: 'Anthropic', color: '#8B5CF6' },
		[ProviderType.OPENAI]: { name: 'OpenAI', color: '#10B981' },
		[ProviderType.GOOGLE]: { name: 'Google AI', color: '#3B82F6' }
	};
	
	// Calculate time series data
	let timeSeriesData = $derived(() => {
		const now = new Date();
		const ranges = {
			'24h': 24,
			'7d': 7 * 24,
			'30d': 30 * 24
		};
		const hours = ranges[selectedTimeRange];
		
		// Generate time points
		const timePoints: Date[] = [];
		for (let i = hours; i >= 0; i -= Math.max(1, Math.floor(hours / 24))) {
			const date = new Date(now.getTime() - i * 60 * 60 * 1000);
			timePoints.push(date);
		}
		
		// Mock time series data (in production, this would come from the backend)
		return stats.map(stat => ({
			provider: stat.provider_type,
			data: timePoints.map(date => ({
				time: date,
				value: Math.random() * 100 // Mock data
			}))
		}));
	});
	
	// Calculate aggregated metrics
	let aggregatedMetrics = $derived(() => {
		const total = {
			requests: 0,
			successful: 0,
			failed: 0,
			tokens: 0,
			cost: 0,
			avgLatency: 0
		};
		
		stats.forEach(stat => {
			total.requests += stat.total_requests;
			total.successful += stat.successful_requests;
			total.failed += stat.failed_requests;
			total.tokens += stat.total_input_tokens + stat.total_output_tokens;
			total.cost += stat.total_cost;
			total.avgLatency += stat.avg_latency_ms * stat.total_requests;
		});
		
		if (total.requests > 0) {
			total.avgLatency /= total.requests;
		}
		
		return total;
	});
	
	// Calculate success rate
	let successRate = $derived(
		aggregatedMetrics.requests > 0
			? (aggregatedMetrics.successful / aggregatedMetrics.requests) * 100
			: 0
	);
	
	// Get top performing provider
	let topProvider = $derived(() => {
		if (stats.length === 0) return null;
		
		return stats.reduce((best, current) => {
			const currentScore = (current.successful_requests / current.total_requests) * 100;
			const bestScore = (best.successful_requests / best.total_requests) * 100;
			return currentScore > bestScore ? current : best;
		});
	});
	
	// Format functions
	function formatNumber(num: number): string {
		if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
		if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
		return num.toLocaleString();
	}
	
	function formatCurrency(amount: number): string {
		return `$${amount.toFixed(2)}`;
	}
	
	function formatLatency(ms: number): string {
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}
	
	function formatPercentage(value: number): string {
		return `${value.toFixed(1)}%`;
	}
	
	function formatDate(date: Date): string {
		return date.toLocaleDateString('en-US', { 
			month: 'short', 
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}
	
	// Get metric value for display
	function getMetricValue(stat: AIProviderStats): number {
		switch (selectedMetric) {
			case 'requests':
				return stat.total_requests;
			case 'tokens':
				return stat.total_input_tokens + stat.total_output_tokens;
			case 'cost':
				return stat.total_cost;
			case 'latency':
				return stat.avg_latency_ms;
			default:
				return 0;
		}
	}
	
	// Format metric value
	function formatMetricValue(value: number): string {
		switch (selectedMetric) {
			case 'requests':
				return formatNumber(value);
			case 'tokens':
				return formatNumber(value);
			case 'cost':
				return formatCurrency(value);
			case 'latency':
				return formatLatency(value);
			default:
				return value.toString();
		}
	}
	
	// Refresh data
	async function refreshData() {
		const endDate = new Date();
		let startDate: Date;
		
		switch (selectedTimeRange) {
			case '24h':
				startDate = new Date(endDate.getTime() - 24 * 60 * 60 * 1000);
				break;
			case '7d':
				startDate = new Date(endDate.getTime() - 7 * 24 * 60 * 60 * 1000);
				break;
			case '30d':
			default:
				startDate = new Date(endDate.getTime() - 30 * 24 * 60 * 60 * 1000);
		}
		
		await providerStore.refreshStats(
			startDate.toISOString(),
			endDate.toISOString()
		);
	}
	
	// Auto-refresh every minute
	$effect(() => {
		const interval = setInterval(refreshData, 60000);
		return () => clearInterval(interval);
	});
</script>

<div class="usage-analytics">
	<!-- Header -->
	<div class="analytics-header">
		<h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">Usage Analytics</h2>
		
		<div class="flex items-center gap-4">
			<!-- Metric Selector -->
			<select 
				bind:value={selectedMetric}
				class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
				       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
			>
				<option value="requests">Requests</option>
				<option value="tokens">Tokens</option>
				<option value="cost">Cost</option>
				<option value="latency">Latency</option>
			</select>
			
			<!-- Time Range Selector -->
			<div class="time-selector">
				<button
					onclick={() => selectedTimeRange = '24h'}
					class="time-button {selectedTimeRange === '24h' ? 'active' : ''}"
				>
					24H
				</button>
				<button
					onclick={() => selectedTimeRange = '7d'}
					class="time-button {selectedTimeRange === '7d' ? 'active' : ''}"
				>
					7D
				</button>
				<button
					onclick={() => selectedTimeRange = '30d'}
					class="time-button {selectedTimeRange === '30d' ? 'active' : ''}"
				>
					30D
				</button>
			</div>
			
			<!-- Refresh Button -->
			<button
				onclick={refreshData}
				class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
				       hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
				title="Refresh data"
			>
				🔄
			</button>
		</div>
	</div>
	
	<!-- Overview Cards -->
	<div class="overview-cards">
		<!-- Total Requests -->
		<div class="metric-card">
			<div class="metric-icon">📊</div>
			<div class="metric-content">
				<div class="metric-label">Total Requests</div>
				<div class="metric-value">{formatNumber(aggregatedMetrics.requests)}</div>
				<div class="metric-change text-green-600">
					{formatPercentage(successRate)} success rate
				</div>
			</div>
		</div>
		
		<!-- Token Usage -->
		<div class="metric-card">
			<div class="metric-icon">🎯</div>
			<div class="metric-content">
				<div class="metric-label">Token Usage</div>
				<div class="metric-value">{formatNumber(aggregatedMetrics.tokens)}</div>
				<div class="metric-change text-blue-600">
					{formatNumber(aggregatedMetrics.tokens / Math.max(1, aggregatedMetrics.requests))} avg/request
				</div>
			</div>
		</div>
		
		<!-- Total Cost -->
		<div class="metric-card">
			<div class="metric-icon">💰</div>
			<div class="metric-content">
				<div class="metric-label">Total Cost</div>
				<div class="metric-value">{formatCurrency(aggregatedMetrics.cost)}</div>
				<div class="metric-change text-orange-600">
					{formatCurrency(aggregatedMetrics.cost / Math.max(1, aggregatedMetrics.requests))} avg/request
				</div>
			</div>
		</div>
		
		<!-- Average Latency -->
		<div class="metric-card">
			<div class="metric-icon">⚡</div>
			<div class="metric-content">
				<div class="metric-label">Avg Latency</div>
				<div class="metric-value">{formatLatency(aggregatedMetrics.avgLatency)}</div>
				{#if topProvider}
					<div class="metric-change text-purple-600">
						Best: {providerConfig[topProvider.provider_type].name}
					</div>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Chart Placeholder -->
	<div class="chart-section">
		<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
			{selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)} Over Time
		</h3>
		
		<div class="chart-placeholder">
			<p class="text-gray-500 dark:text-gray-400">
				Chart visualization would go here
			</p>
			<p class="text-sm text-gray-400 dark:text-gray-500 mt-2">
				Integrate with a charting library like Chart.js or D3.js
			</p>
		</div>
	</div>
	
	<!-- Provider Comparison -->
	<div class="comparison-section">
		<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
			Provider Comparison
		</h3>
		
		<div class="comparison-table">
			<table class="w-full">
				<thead>
					<tr class="border-b border-gray-200 dark:border-gray-700">
						<th class="text-left py-2">Provider</th>
						<th class="text-right py-2">Requests</th>
						<th class="text-right py-2">Success Rate</th>
						<th class="text-right py-2">Avg Latency</th>
						<th class="text-right py-2">Total Cost</th>
						<th class="text-right py-2">Uptime</th>
					</tr>
				</thead>
				<tbody>
					{#each stats as stat}
						{@const providerHealth = health.find(h => h.provider_type === stat.provider_type)}
						<tr class="border-b border-gray-100 dark:border-gray-800">
							<td class="py-3">
								<div class="flex items-center gap-2">
									<div 
										class="w-3 h-3 rounded-full"
										style="background-color: {providerConfig[stat.provider_type].color}"
									></div>
									<span class="font-medium">{providerConfig[stat.provider_type].name}</span>
								</div>
							</td>
							<td class="text-right py-3">{formatNumber(stat.total_requests)}</td>
							<td class="text-right py-3">
								<span class="text-green-600 dark:text-green-400">
									{formatPercentage((stat.successful_requests / stat.total_requests) * 100)}
								</span>
							</td>
							<td class="text-right py-3">{formatLatency(stat.avg_latency_ms)}</td>
							<td class="text-right py-3">{formatCurrency(stat.total_cost)}</td>
							<td class="text-right py-3">
								{#if providerHealth}
									<span class="text-blue-600 dark:text-blue-400">
										{formatPercentage(providerHealth.uptime_percentage)}
									</span>
								{:else}
									-
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
	
	<!-- Performance Insights -->
	<div class="insights-section">
		<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
			Performance Insights
		</h3>
		
		<div class="insights-grid">
			<!-- Best Performing Provider -->
			{#if topProvider}
				<div class="insight-card">
					<div class="insight-icon">🏆</div>
					<div>
						<div class="insight-title">Best Performance</div>
						<div class="insight-value">{providerConfig[topProvider.provider_type].name}</div>
						<div class="insight-detail">
							{formatPercentage((topProvider.successful_requests / topProvider.total_requests) * 100)} success rate
						</div>
					</div>
				</div>
			{/if}
			
			<!-- Cost Efficiency -->
			{#if stats.length > 0}
				{@const mostEfficient = stats.reduce((best, current) => {
					const currentEfficiency = current.total_requests / Math.max(0.01, current.total_cost);
					const bestEfficiency = best.total_requests / Math.max(0.01, best.total_cost);
					return currentEfficiency > bestEfficiency ? current : best;
				})}
				<div class="insight-card">
					<div class="insight-icon">💡</div>
					<div>
						<div class="insight-title">Most Cost-Effective</div>
						<div class="insight-value">{providerConfig[mostEfficient.provider_type].name}</div>
						<div class="insight-detail">
							{formatCurrency(mostEfficient.total_cost / Math.max(1, mostEfficient.total_requests))} per request
						</div>
					</div>
				</div>
			{/if}
			
			<!-- Fastest Response -->
			{#if stats.length > 0}
				{@const fastest = stats.reduce((best, current) => 
					current.avg_latency_ms < best.avg_latency_ms ? current : best
				)}
				<div class="insight-card">
					<div class="insight-icon">⚡</div>
					<div>
						<div class="insight-title">Fastest Response</div>
						<div class="insight-value">{providerConfig[fastest.provider_type].name}</div>
						<div class="insight-detail">
							{formatLatency(fastest.avg_latency_ms)} average
						</div>
					</div>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.usage-analytics {
		@apply space-y-6;
	}
	
	.analytics-header {
		@apply flex items-center justify-between;
	}
	
	.time-selector {
		@apply inline-flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1;
	}
	
	.time-button {
		@apply px-3 py-1 rounded-md text-sm font-medium text-gray-600 dark:text-gray-400
		       hover:text-gray-900 dark:hover:text-gray-100 transition-colors;
	}
	
	.time-button.active {
		@apply bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm;
	}
	
	.overview-cards {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4;
	}
	
	.metric-card {
		@apply flex items-center gap-4 p-4 bg-white dark:bg-gray-900 rounded-lg 
		       shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.metric-icon {
		@apply text-2xl;
	}
	
	.metric-label {
		@apply text-sm font-medium text-gray-600 dark:text-gray-400;
	}
	
	.metric-value {
		@apply text-2xl font-bold text-gray-900 dark:text-gray-100;
	}
	
	.metric-change {
		@apply text-sm mt-1;
	}
	
	.chart-section {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.chart-placeholder {
		@apply h-64 bg-gray-50 dark:bg-gray-800 rounded-lg flex items-center justify-center flex-col;
	}
	
	.comparison-section {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.comparison-table {
		@apply overflow-x-auto;
	}
	
	.comparison-table table {
		@apply text-sm;
	}
	
	.comparison-table th {
		@apply text-gray-600 dark:text-gray-400 font-medium;
	}
	
	.comparison-table td {
		@apply text-gray-900 dark:text-gray-100;
	}
	
	.insights-section {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.insights-grid {
		@apply grid grid-cols-1 md:grid-cols-3 gap-4;
	}
	
	.insight-card {
		@apply flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg;
	}
	
	.insight-icon {
		@apply text-2xl;
	}
	
	.insight-title {
		@apply text-sm font-medium text-gray-600 dark:text-gray-400;
	}
	
	.insight-value {
		@apply text-lg font-semibold text-gray-900 dark:text-gray-100;
	}
	
	.insight-detail {
		@apply text-sm text-gray-500 dark:text-gray-400;
	}
</style>