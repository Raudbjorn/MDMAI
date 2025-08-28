// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
	namespace App {
		interface Error {
			message: string;
			code?: string;
		}
		interface Locals {
			user?: {
				id: string;
				username: string;
				role: 'player' | 'gm' | 'admin';
			};
			session?: {
				id: string;
				campaignId?: string;
			};
		}
		interface PageData {}
		interface PageState {}
		interface Platform {}
	}
}

export {};