import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Building Production-Grade Multi-Agent RAG Systems',
  tagline: 'A Comprehensive Guide to ResearcherAI - From Concept to Production',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://sebuliba-adrian.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/ResearcherAI-Book/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Sebuliba-Adrian', // Usually your GitHub org/user name.
  projectName: 'ResearcherAI-Book', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Sebuliba-Adrian/ResearcherAI-Book/tree/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'ResearcherAI Book',
      logo: {
        alt: 'ResearcherAI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'bookSidebar',
          position: 'left',
          label: '📚 Read the Book',
        },
        {
          href: 'https://github.com/Sebuliba-Adrian/ResearcherAI',
          label: 'Source Code',
          position: 'right',
        },
        {
          href: 'https://github.com/Sebuliba-Adrian/ResearcherAI-Book',
          label: 'Book Repository',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book Content',
          items: [
            {
              label: 'Introduction',
              to: '/',
            },
            {
              label: 'Primer for Web Developers',
              to: '/primer',
            },
            {
              label: 'Glossary',
              to: '/glossary',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'ResearcherAI Source Code',
              href: 'https://github.com/Sebuliba-Adrian/ResearcherAI',
            },
            {
              label: 'Bibliography',
              to: '/bibliography',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Report Issue',
              href: 'https://github.com/Sebuliba-Adrian/ResearcherAI-Book/issues',
            },
            {
              label: 'Book Repository',
              href: 'https://github.com/Sebuliba-Adrian/ResearcherAI-Book',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Adrian Sebuliba. All rights reserved.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
