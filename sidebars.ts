import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  bookSidebar: [
    {
      type: 'category',
      label: '📖 Front Matter',
      items: ['frontmatter'],
    },
    {
      type: 'category',
      label: '🎓 Getting Started',
      items: ['primer', 'glossary'],
    },
    {
      type: 'doc',
      id: 'intro',
      label: '🎯 Introduction',
    },
    {
      type: 'category',
      label: '📚 Main Chapters',
      items: [
        'planning',
        'architecture',
        'frameworks',
        'backend',
        'frontend',
        'testing',
      ],
    },
    {
      type: 'category',
      label: '🚀 Production Deployment',
      items: [
        'deployment',
        'kubernetes',
        'terraform',
      ],
    },
    {
      type: 'category',
      label: '📊 Operations & Monitoring',
      items: [
        'observability',
        'cicd',
        'monitoring',
        'conclusion',
      ],
    },
    {
      type: 'category',
      label: '📚 Reference',
      items: ['bibliography'],
    },
  ],
};

export default sidebars;
