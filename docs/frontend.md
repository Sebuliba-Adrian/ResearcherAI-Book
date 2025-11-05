---
layout: default
title: Frontend Development
---

# Frontend Development


With a solid backend in place, it was time to build a beautiful, modern UI. I wanted something that felt professional, responsive, and delightful to use.


## Design Vision

I had a clear vision:
- **Glassmorphism aesthetic**: Frosted glass effects with subtle blur
- **Smooth animations**: 60fps transitions using Framer Motion
- **Fully responsive**: Mobile-first design
- **Type-safe**: TypeScript for confidence
- **Fast**: Vite for instant HMR

## Setting Up the Stack

```bash
# Create Vite project with React + TypeScript
npm create vite@latest frontend-react -- --template react-ts

cd frontend-react

# Install dependencies
npm install

# Add Tailwind CSS
npm install -D tailwindcss@3.4.0 postcss autoprefixer
npx tailwindcss init -p

# Add routing and API client
npm install react-router-dom axios

# Add animations and icons
npm install framer-motion lucide-react
```

### Why These Choices?

**Vite over Create React App**
- ⚡ Lightning-fast HMR (< 1s vs 5-10s)
- 📦 Smaller bundle size
- 🛠️ Better TypeScript support
- 🎯 Modern ESM-based architecture

**Tailwind v3 over v4**
- ✅ Stable and battle-tested
- ✅ Great documentation
- ✅ Supports `@apply` for custom components
- ⚠️ v4 had breaking changes I didn't want to deal with

**Framer Motion over CSS animations**
- 🎭 Declarative animation API
- 🎯 Physics-based springs
- 📱 Touch gestures built-in
- 🔄 Easy to orchestrate complex sequences

## Project Structure

```
frontend-react/
├── src/
│   ├── components/
│   │   ├── Common/
│   │   │   ├── GlassCard.tsx
│   │   │   ├── Toast.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   ├── Layout/
│   │   │   ├── Navigation.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Footer.tsx
│   │   ├── DataCollection/
│   │   │   ├── SourceSelector.tsx
│   │   │   ├── QueryInput.tsx
│   │   │   └── ResultsGrid.tsx
│   │   ├── Graph/
│   │   │   └── GraphVisualization.tsx
│   │   ├── Query/
│   │   │   ├── ChatInterface.tsx
│   │   │   └── MessageBubble.tsx
│   │   ├── Sessions/
│   │   │   └── SessionCard.tsx
│   │   ├── Upload/
│   │   │   └── DragDropZone.tsx
│   │   └── Vector/
│   │       └── EmbeddingPlot.tsx
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Collect.tsx
│   │   ├── Ask.tsx
│   │   ├── Graph.tsx
│   │   ├── Vector.tsx
│   │   ├── Upload.tsx
│   │   └── Sessions.tsx
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Glassmorphism Design System

First, I set up the design system in Tailwind config and global CSS.

### Tailwind Configuration

```javascript
// tailwind.config.js
export default {
  content: ['./intro', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          400: '#0ea5e9',
          500: '#0070f3',
          600: '#005bb5',
        },
        neon: {
          purple: '#a855f7',
          pink: '#ec4899',
          blue: '#3b82f6',
        }
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        glow: 'glow 2s ease-in-out infinite',
        shimmer: 'shimmer 2s linear infinite'
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' }
        },
        glow: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' }
        },
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' }
        }
      }
    }
  }
}
```

### Global Styles

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100;
    min-height: 100vh;
  }
}

@layer components {
  /* Glass card effect */
  .glass-card {
    @apply bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl shadow-xl p-6;
    @apply transition-all duration-300 hover:bg-white/15 hover:shadow-2xl;
  }

  /* Glass buttons */
  .btn-glass-primary {
    @apply bg-gradient-to-r from-cyan-500 to-purple-600 text-white px-6 py-3 rounded-2xl;
    @apply font-semibold transition-all duration-300;
    @apply hover:shadow-2xl hover:shadow-purple-500/50 hover:scale-105 active:scale-95;
  }

  .btn-glass-secondary {
    @apply bg-white/5 backdrop-blur-sm border border-white/10 px-6 py-3 rounded-2xl;
    @apply text-slate-200 hover:bg-white/10 hover:border-white/20;
    @apply transition-all duration-300;
  }

  /* Glass input */
  .input-glass {
    @apply bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl px-4 py-3;
    @apply text-slate-100 placeholder-slate-400;
    @apply focus:bg-white/10 focus:border-cyan-500/50 focus:outline-none;
    @apply transition-all duration-300;
  }
}

/* Animated orbs in background */
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(40px);
  opacity: 0.6;
  animation: blob 7s infinite;
}

@keyframes blob {
  0%, 100% { transform: translate(0px, 0px) scale(1); }
  33% { transform: translate(30px, -50px) scale(1.1); }
  66% { transform: translate(-20px, 20px) scale(0.9); }
}
```

## Building Pages

### Home Page with Animated Hero

```typescript
// src/pages/Home.tsx
import { motion } from 'framer-motion';
import { Database, Brain, Network, Zap, Search, Upload } from 'lucide-react';

const Home = () => {
  const agents = [
    {
      icon: Database,
      title: 'Data Collector',
      description: 'Aggregates information from 7 diverse sources',
      color: 'from-blue-500 to-cyan-500',
      stats: '7 Sources'
    },
    {
      icon: Network,
      title: 'Knowledge Graph',
      description: 'Builds relationships between papers, authors, topics',
      color: 'from-purple-500 to-pink-500',
      stats: 'Neo4j/NetworkX'
    },
    {
      icon: Search,
      title: 'Vector Search',
      description: 'Semantic search with 384-dimensional embeddings',
      color: 'from-green-500 to-teal-500',
      stats: 'Qdrant/FAISS'
    },
    // ... more agents
  ];

  return (
    
      {/* Animated background orbs */}
      <motion.div
        className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"
        animate={{
          scale: [1, 1.2, 1],
          x: [0, 50, 0],
          y: [0, 30, 0],
        }}
        transition={{ duration: 8, repeat: Infinity }}
      />

      <motion.div
        className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl"
        animate={{
          scale: [1.2, 1, 1.2],
          x: [0, -30, 0],
          y: [0, 50, 0],
        }}
        transition={{ duration: 10, repeat: Infinity }}
      />

      {/* Hero section */}
      
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-7xl font-bold mb-6 bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            ResearcherAI
          </h1>
          <p className="text-2xl text-slate-300 mb-8">
            Multi-Agent RAG System for Research Paper Analysis
          </p>

          
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-glass-primary"
            >
              Get Started
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-glass-secondary"
            >
              Learn More
            </motion.button>
          
        </motion.div>

        {/* Agent cards */}
        
          {agents.map((agent, index) => (
            <motion.div
              key={agent.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-card group cursor-pointer"
            >
              
                <agent.icon className="w-full h-full text-white" />
              

              <h3 className="text-xl font-bold mb-2">{agent.title}</h3>
              <p className="text-slate-400 mb-4">{agent.description}</p>

              
                {agent.stats}
              
            </motion.div>
          ))}
        
      
    
  );
};

export default Home;
```

### Data Collection Page

```typescript
// src/pages/Collect.tsx
import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Loader2 } from 'lucide-react';
import { api } from '../services/api';
import type { Paper } from '../types';

const Collect = () => {
  const [query, setQuery] = useState('');
  const [sources, setSources] = useState<string[]>(['arxiv', 'semantic_scholar']);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Paper[]>([]);

  const availableSources = [
    { id: 'arxiv', name: 'arXiv' },
    { id: 'semantic_scholar', name: 'Semantic Scholar' },
    { id: 'pubmed', name: 'PubMed' },
    { id: 'zenodo', name: 'Zenodo' },
    { id: 'web', name: 'Web Search' },
    { id: 'huggingface', name: 'HuggingFace' },
    { id: 'kaggle', name: 'Kaggle' },
  ];

  const handleCollect = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await api.collectData({
        query,
        sources,
        max_per_source: 10,
      });

      setResults(response.papers);
    } catch (error) {
      console.error('Collection failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    
      <h1 className="text-4xl font-bold mb-8">Collect Research Papers</h1>

      {/* Query input */}
      
        <label className="block text-sm font-medium mb-2">Search Query</label>
        
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., transformer neural networks"
            className="input-glass flex-1"
            onKeyPress={(e) => e.key === 'Enter' && handleCollect()}
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleCollect}
            disabled={loading}
            className="btn-glass-primary min-w-[120px]"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin mx-auto" />
            ) : (
              <>
                <Search className="w-5 h-5 inline mr-2" />
                Search
              </>
            )}
          </motion.button>
        
      

      {/* Source selector */}
      
        <label className="block text-sm font-medium mb-4">Data Sources</label>
        
          {availableSources.map((source) => (
            <motion.label
              key={source.id}
              whileHover={{ scale: 1.02 }}
              className={`
                relative flex items-center gap-3 p-3 rounded-xl cursor-pointer
                transition-all duration-300
                ${sources.includes(source.id)
                  ? 'bg-cyan-500/20 border-2 border-cyan-500'
                  : 'bg-white/5 border-2 border-transparent hover:bg-white/10'
                }
              `}
            >
              <input
                type="checkbox"
                checked={sources.includes(source.id)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSources([...sources, source.id]);
                  } else {
                    setSources(sources.filter(s => s !== source.id));
                  }
                }}
                className="sr-only"
              />
              <span className="text-sm font-medium">{source.name}</span>
            </motion.label>
          ))}
        
      

      {/* Results */}
      {results.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          <h2 className="text-2xl font-bold mb-4">
            Found {results.length} Papers
          </h2>

          {results.map((paper, index) => (
            <motion.div
              key={paper.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="glass-card"
            >
              <h3 className="text-xl font-bold mb-2">{paper.title}</h3>
              <p className="text-slate-400 mb-3">
                {paper.authors.slice(0, 3).join(', ')}
                {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
              </p>
              <p className="text-sm text-slate-300 mb-4">
                {paper.abstract.slice(0, 300)}...
              </p>

              
                <span className="bg-purple-500/20 px-3 py-1 rounded-full text-sm">
                  {paper.source}
                </span>
                <span className="bg-cyan-500/20 px-3 py-1 rounded-full text-sm">
                  {paper.published_date}
                </span>
              
            </motion.div>
          ))}
        </motion.div>
      )}
    
  );
};

export default Collect;
```

### Chat Interface (Ask Page)

```typescript
// src/pages/Ask.tsx
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Loader2, User, Bot } from 'lucide-react';
import { api } from '../services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

const Ask = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.query({
        question: input,
        session_id: 'default',
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        timestamp: Date.now(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    
      {/* Messages */}
      
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                
                  <Bot className="w-6 h-6 text-white" />
                
              )}

              <div
                className={`
                  max-w-2xl p-4 rounded-2xl
                  ${message.role === 'user'
                    ? 'bg-gradient-to-r from-cyan-500 to-purple-600 text-white'
                    : 'glass-card'
                  }
                `}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              

              {message.role === 'user' && (
                
                  <User className="w-6 h-6 text-white" />
                
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-4"
          >
            
              <Bot className="w-6 h-6 text-white" />
            
            
              <Loader2 className="w-5 h-5 animate-spin" />
            
          </motion.div>
        )}

        
      

      {/* Input */}
      
        
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="Ask a question about your research..."
            className="input-glass flex-1"
            disabled={loading}
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="btn-glass-primary"
          >
            <Send className="w-5 h-5" />
          </motion.button>
        
      
    
  );
};

export default Ask;
```

## API Integration Layer

```typescript
// src/services/api.ts
import axios, { AxiosInstance } from 'axios';

class APIService {
  private api: AxiosInstance;

  constructor() {
    const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    this.api = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        throw error;
      }
    );
  }

  async collectData(data: {
    query: string;
    sources: string[];
    max_per_source: number;
  }) {
    const response = await this.api.post('/api/collect', data);
    return response.data;
  }

  async query(data: { question: string; session_id: string }) {
    const response = await this.api.post('/api/query', data);
    return response.data;
  }

  async getGraph(session_id: string) {
    const response = await this.api.get(`/api/graph/${session_id}`);
    return response.data;
  }

  async searchVectors(query: string, top_k: number = 10) {
    const response = await this.api.post('/api/vector/search', {
      query,
      top_k,
    });
    return response.data;
  }

  async uploadFile(file: File, session_id: string) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', session_id);

    const response = await this.api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getSessions() {
    const response = await this.api.get('/api/sessions');
    return response.data;
  }
}

export const api = new APIService();
```

## TypeScript Types

```typescript
// src/types/index.ts
export interface Paper {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published_date: string;
  source: string;
  url: string;
  citations?: number;
}

export interface Session {
  id: string;
  name: string;
  created_at: string;
  papers_collected: number;
  conversations: Conversation[];
  metadata: Record<string, any>;
}

export interface Conversation {
  question: string;
  answer: string;
  sources: string[];
  timestamp: string;
}

export interface GraphNode {
  id: string;
  label: string;
  properties: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relationship: string;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  metadata: {
    paper_id: string;
    title: string;
    text: string;
    chunk_index: number;
  };
}
```

## What I Learned

**✅ Wins**

1. **Vite is incredibly fast**: HMR in < 1 second
2. **Glassmorphism looks professional**: Users love the aesthetic
3. **Framer Motion is powerful**: Declarative animations are easy
4. **TypeScript catches bugs early**: Worth the setup time
5. **Tailwind speeds development**: No CSS files to manage

**🤔 Challenges**

1. **Tailwind v4 breaking changes**: Had to downgrade to v3
2. **Animation performance on mobile**: Needed to reduce blur effects
3. **Type safety with API responses**: Required careful type definitions
4. **Glassmorphism with dark mode**: Tricky to get contrast right

**💡 Insights**

> Good design takes time, but it pays off in user satisfaction.

> Animations should enhance, not distract. Keep them subtle.

> Type safety is worth the initial overhead - saves debugging time later.

## Next: Testing

Frontend done! Now let's make sure everything works with comprehensive tests.


  <a href="backend">← Back: Backend</a>
  <a href="testing">Next: Testing Strategy →</a>

