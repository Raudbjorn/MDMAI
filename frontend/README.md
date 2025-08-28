# TTRPG Assistant Frontend

A modern, responsive web interface for the TTRPG Assistant MCP Server built with SvelteKit.

## Features

- **SvelteKit Framework**: Server-side rendering with optimal performance
- **TypeScript**: Full type safety throughout the application
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Updates**: WebSocket and Server-Sent Events for live collaboration
- **Modern UI Components**: Built with bits-ui and Tailwind CSS
- **Dark Mode Support**: Automatic theme detection with mode-watcher

## Tech Stack

- **Framework**: SvelteKit with adapter-node for SSR
- **Language**: TypeScript
- **Styling**: TailwindCSS with custom design tokens
- **Components**: bits-ui (Svelte alternative to shadcn/ui)
- **State Management**: Native Svelte 5 runes and stores
- **Build Tool**: Vite
- **Icons**: lucide-svelte

## Project Structure

```
frontend/
├── src/
│   ├── app.html           # HTML template
│   ├── app.css            # Global styles and Tailwind imports
│   ├── app.d.ts           # TypeScript app definitions
│   ├── lib/
│   │   ├── api/           # API client and types
│   │   ├── components/    # UI components
│   │   ├── stores/        # Svelte stores for state
│   │   └── utils/         # Utility functions
│   └── routes/
│       ├── +layout.svelte # Root layout
│       ├── +page.svelte   # Home page
│       └── dashboard/     # Dashboard route
├── static/                # Static assets
├── package.json          # Dependencies
├── svelte.config.js      # SvelteKit configuration
├── vite.config.ts        # Vite configuration
├── tailwind.config.js    # Tailwind configuration
└── tsconfig.json         # TypeScript configuration
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or pnpm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# The app will be available at http://localhost:5173
```

### Building for Production

```bash
# Build the application
npm run build

# Preview production build
npm run preview

# Run production server
node build
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run check` - Type check the project
- `npm run lint` - Lint the code
- `npm run format` - Format code with Prettier

## Components

### Core UI Components

- **Button**: Multiple variants (default, destructive, outline, secondary, ghost, link)
- **Card**: Container with header, title, description, and content sections
- **Tabs**: Tab navigation with content panels
- **Toast**: Notifications via svelte-sonner

### API Integration

The frontend connects to the MCP bridge service via:
- WebSocket for bidirectional real-time communication
- Server-Sent Events for server-pushed updates
- REST API for standard CRUD operations

### State Management

Using Svelte 5's new runes system:
- `$state` for reactive state
- `$derived` for computed values
- `$effect` for side effects
- Native stores for global state

## Environment Variables

Create a `.env` file in the frontend directory:

```env
PUBLIC_API_URL=http://localhost:8000
PUBLIC_WS_URL=ws://localhost:8000
```

## Development Notes

- All components use Svelte 5 runes syntax
- TypeScript strict mode is enabled
- Tailwind classes are merged using tailwind-merge
- Design tokens follow a consistent naming convention
- Server-side rendering is enabled by default

## Contributing

1. Follow the existing code style
2. Use TypeScript for all new code
3. Ensure components are accessible
4. Test on multiple screen sizes
5. Update documentation as needed

## License

See the main project LICENSE file.