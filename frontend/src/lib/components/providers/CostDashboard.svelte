<script lang="ts">
	import { providerStore } from '$lib/stores/providers.svelte';
	import { ProviderType } from '$lib/types/providers';
	import type { CostBudget, AIProviderStats } from '$lib/types/providers';
	
	// Store data
	let stats = $derived(providerStore.stats);
	let budgets = $derived(providerStore.budgets);
	let totalCost = $derived(providerStore.totalCost);
	
	// Local state
	let selectedPeriod = $state<'day' | 'week' | 'month'>('month');
	let selectedProvider = $state<ProviderType | 'all'>('all');
	let showBudgetModal = $state(false);
	let editingBudget = $state<CostBudget | null>(null);
	
	// Budget form state
	let budgetForm = $state({
		name: '',
		daily_limit: 0,
		monthly_limit: 0,
		alert_thresholds: [0.5, 0.8, 0.95]
	});
	
	// Provider colors for charts
	const providerColors: Record<ProviderType, string> = {
		[ProviderType.ANTHROPIC]: '#8B5CF6', // Purple
		[ProviderType.OPENAI]: '#10B981', // Green
		[ProviderType.GOOGLE]: '#3B82F6' // Blue
	};
	
	// Calculate period costs
	let periodCosts = $derived(() => {
		const now = new Date();
		let startDate: Date;
		
		switch (selectedPeriod) {
			case 'day':
				startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
				break;
			case 'week':
				startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
				break;
			case 'month':
			default:
				startDate = new Date(now.getFullYear(), now.getMonth(), 1);
		}
		
		return stats
			.filter(s => selectedProvider === 'all' || s.provider_type === selectedProvider)
			.reduce((total, stat) => {
				// Filter costs based on period
				const periodKey = startDate.toISOString().split('T')[0];
				return total + (stat.daily_usage[periodKey] || 0);
			}, 0);
	});
	
	// Calculate budget usage
	let budgetUsage = $derived(() => {
		if (budgets.length === 0) return null;
		
		const activeBudget = budgets.find(b => b.enabled);
		if (!activeBudget) return null;
		
		const used = periodCosts;
		const limit = selectedPeriod === 'day' 
			? activeBudget.daily_limit 
			: activeBudget.monthly_limit;
		
		if (!limit) return null;
		
		return {
			used,
			limit,
			percentage: (used / limit) * 100,
			remaining: limit - used,
			budget: activeBudget
		};
	});
	
	// Get provider stats
	function getProviderStats(type: ProviderType): AIProviderStats | undefined {
		return stats.find(s => s.provider_type === type);
	}
	
	// Format currency
	function formatCurrency(amount: number): string {
		return new Intl.NumberFormat('en-US', {
			style: 'currency',
			currency: 'USD',
			minimumFractionDigits: 2,
			maximumFractionDigits: 2
		}).format(amount);
	}
	
	// Format percentage
	function formatPercentage(value: number): string {
		return `${value.toFixed(1)}%`;
	}
	
	// Format number with commas
	function formatNumber(num: number): string {
		return num.toLocaleString();
	}
	
	// Get budget alert level
	function getBudgetAlertLevel(percentage: number): 'safe' | 'warning' | 'danger' | 'exceeded' {
		if (percentage >= 100) return 'exceeded';
		if (percentage >= 95) return 'danger';
		if (percentage >= 80) return 'warning';
		return 'safe';
	}
	
	// Save budget
	async function saveBudget() {
		const budget: CostBudget = {
			budget_id: editingBudget?.budget_id || crypto.randomUUID(),
			name: budgetForm.name,
			daily_limit: budgetForm.daily_limit || undefined,
			monthly_limit: budgetForm.monthly_limit || undefined,
			provider_limits: {},
			alert_thresholds: budgetForm.alert_thresholds,
			enabled: true,
			created_at: editingBudget?.created_at || new Date()
		};
		
		try {
			await providerStore.configureBudget(budget);
			showBudgetModal = false;
			resetBudgetForm();
		} catch (error) {
			console.error('Failed to save budget:', error);
		}
	}
	
	// Delete budget
	async function deleteBudget(budgetId: string) {
		if (!confirm('Are you sure you want to delete this budget?')) return;
		
		try {
			await providerStore.removeBudget(budgetId);
		} catch (error) {
			console.error('Failed to delete budget:', error);
		}
	}
	
	// Edit budget
	function editBudget(budget: CostBudget) {
		editingBudget = budget;
		budgetForm = {
			name: budget.name,
			daily_limit: budget.daily_limit || 0,
			monthly_limit: budget.monthly_limit || 0,
			alert_thresholds: budget.alert_thresholds
		};
		showBudgetModal = true;
	}
	
	// Reset budget form
	function resetBudgetForm() {
		editingBudget = null;
		budgetForm = {
			name: '',
			daily_limit: 0,
			monthly_limit: 0,
			alert_thresholds: [0.5, 0.8, 0.95]
		};
	}
	
	// Calculate cost breakdown by provider
	let providerBreakdown = $derived(() => {
		const breakdown: Record<ProviderType, number> = {
			[ProviderType.ANTHROPIC]: 0,
			[ProviderType.OPENAI]: 0,
			[ProviderType.GOOGLE]: 0
		};
		
		stats.forEach(stat => {
			breakdown[stat.provider_type] = stat.total_cost;
		});
		
		return Object.entries(breakdown)
			.filter(([_, cost]) => cost > 0)
			.map(([provider, cost]) => ({
				provider: provider as ProviderType,
				cost,
				percentage: totalCost > 0 ? (cost / totalCost) * 100 : 0
			}));
	});
</script>

<div class="cost-dashboard">
	<!-- Header -->
	<div class="dashboard-header">
		<h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">Cost Management</h2>
		
		<div class="flex items-center gap-4">
			<!-- Period Selector -->
			<div class="period-selector">
				<button
					onclick={() => selectedPeriod = 'day'}
					class="period-button {selectedPeriod === 'day' ? 'active' : ''}"
				>
					Day
				</button>
				<button
					onclick={() => selectedPeriod = 'week'}
					class="period-button {selectedPeriod === 'week' ? 'active' : ''}"
				>
					Week
				</button>
				<button
					onclick={() => selectedPeriod = 'month'}
					class="period-button {selectedPeriod === 'month' ? 'active' : ''}"
				>
					Month
				</button>
			</div>
			
			<!-- Provider Filter -->
			<select 
				bind:value={selectedProvider}
				class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
				       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
			>
				<option value="all">All Providers</option>
				<option value={ProviderType.ANTHROPIC}>Anthropic</option>
				<option value={ProviderType.OPENAI}>OpenAI</option>
				<option value={ProviderType.GOOGLE}>Google AI</option>
			</select>
			
			<!-- Budget Button -->
			<button
				onclick={() => { showBudgetModal = true; resetBudgetForm(); }}
				class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
			>
				+ Add Budget
			</button>
		</div>
	</div>
	
	<!-- Summary Cards -->
	<div class="summary-cards">
		<!-- Total Cost Card -->
		<div class="summary-card">
			<div class="card-label">Total Cost</div>
			<div class="card-value text-2xl font-bold">{formatCurrency(periodCosts)}</div>
			<div class="card-subtitle">
				{selectedPeriod === 'day' ? 'Today' : 
				 selectedPeriod === 'week' ? 'Last 7 days' : 'This month'}
			</div>
		</div>
		
		<!-- Budget Status Card -->
		{#if budgetUsage}
			{@const alertLevel = getBudgetAlertLevel(budgetUsage.percentage)}
			<div class="summary-card budget-card budget-{alertLevel}">
				<div class="card-label">Budget Usage</div>
				<div class="card-value text-2xl font-bold">
					{formatPercentage(budgetUsage.percentage)}
				</div>
				<div class="card-subtitle">
					{formatCurrency(budgetUsage.used)} / {formatCurrency(budgetUsage.limit)}
				</div>
				<div class="budget-bar">
					<div 
						class="budget-fill budget-fill-{alertLevel}"
						style="width: {Math.min(100, budgetUsage.percentage)}%"
					></div>
				</div>
			</div>
		{:else}
			<div class="summary-card">
				<div class="card-label">Budget Status</div>
				<div class="card-value text-lg text-gray-500">No budget set</div>
				<button 
					onclick={() => { showBudgetModal = true; resetBudgetForm(); }}
					class="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 mt-2"
				>
					Configure budget →
				</button>
			</div>
		{/if}
		
		<!-- Request Count Card -->
		<div class="summary-card">
			<div class="card-label">Total Requests</div>
			<div class="card-value text-2xl font-bold">
				{formatNumber(stats.reduce((sum, s) => sum + s.total_requests, 0))}
			</div>
			<div class="card-subtitle">
				Success rate: {formatPercentage(
					(stats.reduce((sum, s) => sum + s.successful_requests, 0) / 
					 stats.reduce((sum, s) => sum + s.total_requests, 0) || 0) * 100
				)}
			</div>
		</div>
		
		<!-- Token Usage Card -->
		<div class="summary-card">
			<div class="card-label">Token Usage</div>
			<div class="card-value text-2xl font-bold">
				{formatNumber(stats.reduce((sum, s) => sum + s.total_input_tokens + s.total_output_tokens, 0))}
			</div>
			<div class="card-subtitle">
				Input: {formatNumber(stats.reduce((sum, s) => sum + s.total_input_tokens, 0))} |
				Output: {formatNumber(stats.reduce((sum, s) => sum + s.total_output_tokens, 0))}
			</div>
		</div>
	</div>
	
	<!-- Provider Breakdown -->
	{#if providerBreakdown.length > 0}
		<div class="breakdown-section">
			<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
				Cost by Provider
			</h3>
			
			<div class="breakdown-list">
				{#each providerBreakdown as item}
					<div class="breakdown-item">
						<div class="flex items-center justify-between mb-2">
							<span class="font-medium">{item.provider}</span>
							<span class="text-sm text-gray-600 dark:text-gray-400">
								{formatCurrency(item.cost)} ({formatPercentage(item.percentage)})
							</span>
						</div>
						<div class="breakdown-bar">
							<div 
								class="breakdown-fill"
								style="width: {item.percentage}%; background-color: {providerColors[item.provider]}"
							></div>
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
	
	<!-- Budget List -->
	{#if budgets.length > 0}
		<div class="budget-section">
			<h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
				Active Budgets
			</h3>
			
			<div class="budget-list">
				{#each budgets as budget}
					<div class="budget-item">
						<div class="flex items-center justify-between">
							<div>
								<h4 class="font-medium">{budget.name}</h4>
								<div class="text-sm text-gray-600 dark:text-gray-400">
									{#if budget.daily_limit}
										Daily: {formatCurrency(budget.daily_limit)}
									{/if}
									{#if budget.monthly_limit}
										{budget.daily_limit ? ' | ' : ''}
										Monthly: {formatCurrency(budget.monthly_limit)}
									{/if}
								</div>
							</div>
							<div class="flex items-center gap-2">
								<button
									onclick={() => editBudget(budget)}
									class="text-blue-600 hover:text-blue-700 dark:text-blue-400"
								>
									Edit
								</button>
								<button
									onclick={() => deleteBudget(budget.budget_id)}
									class="text-red-600 hover:text-red-700 dark:text-red-400"
								>
									Delete
								</button>
							</div>
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

<!-- Budget Modal -->
{#if showBudgetModal}
	<div class="modal-overlay" onclick={() => (showBudgetModal = false)}>
		<div class="modal-content" onclick={(e) => e.stopPropagation()}>
			<h3 class="text-lg font-semibold mb-4">
				{editingBudget ? 'Edit Budget' : 'Add Budget'}
			</h3>
			
			<div class="space-y-4">
				<div>
					<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
						Budget Name
					</label>
					<input
						type="text"
						bind:value={budgetForm.name}
						placeholder="e.g., Monthly AI Budget"
						class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
						       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
					/>
				</div>
				
				<div class="grid grid-cols-2 gap-4">
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
							Daily Limit (USD)
						</label>
						<input
							type="number"
							bind:value={budgetForm.daily_limit}
							min="0"
							step="0.01"
							placeholder="0.00"
							class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
							       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
						/>
					</div>
					
					<div>
						<label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
							Monthly Limit (USD)
						</label>
						<input
							type="number"
							bind:value={budgetForm.monthly_limit}
							min="0"
							step="0.01"
							placeholder="0.00"
							class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
							       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
						/>
					</div>
				</div>
				
				<div class="flex justify-end gap-2 pt-4">
					<button
						onclick={() => showBudgetModal = false}
						class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md 
						       hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
					>
						Cancel
					</button>
					<button
						onclick={saveBudget}
						class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
					>
						{editingBudget ? 'Update' : 'Create'} Budget
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.cost-dashboard {
		@apply space-y-6;
	}
	
	.dashboard-header {
		@apply flex items-center justify-between;
	}
	
	.period-selector {
		@apply inline-flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1;
	}
	
	.period-button {
		@apply px-3 py-1 rounded-md text-sm font-medium text-gray-600 dark:text-gray-400
		       hover:text-gray-900 dark:hover:text-gray-100 transition-colors;
	}
	
	.period-button.active {
		@apply bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm;
	}
	
	.summary-cards {
		@apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4;
	}
	
	.summary-card {
		@apply p-4 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.card-label {
		@apply text-sm font-medium text-gray-600 dark:text-gray-400 mb-1;
	}
	
	.card-value {
		@apply text-gray-900 dark:text-gray-100;
	}
	
	.card-subtitle {
		@apply text-sm text-gray-500 dark:text-gray-400 mt-1;
	}
	
	.budget-bar {
		@apply w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-2 overflow-hidden;
	}
	
	.budget-fill {
		@apply h-full transition-all duration-300;
	}
	
	.budget-fill-safe {
		@apply bg-green-500;
	}
	
	.budget-fill-warning {
		@apply bg-yellow-500;
	}
	
	.budget-fill-danger {
		@apply bg-orange-500;
	}
	
	.budget-fill-exceeded {
		@apply bg-red-500;
	}
	
	.breakdown-section {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.breakdown-list {
		@apply space-y-4;
	}
	
	.breakdown-bar {
		@apply w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
	}
	
	.breakdown-fill {
		@apply h-full transition-all duration-300;
	}
	
	.budget-section {
		@apply p-6 bg-white dark:bg-gray-900 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700;
	}
	
	.budget-list {
		@apply space-y-3;
	}
	
	.budget-item {
		@apply p-3 border border-gray-200 dark:border-gray-700 rounded-md;
	}
	
	.modal-overlay {
		@apply fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50;
	}
	
	.modal-content {
		@apply bg-white dark:bg-gray-900 rounded-lg p-6 max-w-md w-full mx-4;
	}
</style>