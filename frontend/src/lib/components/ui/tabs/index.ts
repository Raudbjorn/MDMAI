import Tabs from './tabs.svelte';
import TabsContent from './tabs-content.svelte';
import TabsList from './tabs-list.svelte';
import TabsTrigger from './tabs-trigger.svelte';

export {
	Tabs,
	TabsContent,
	TabsList,
	TabsTrigger
};

// Re-export types from bits-ui for convenience
export type { Tabs as TabsPrimitive } from 'bits-ui';