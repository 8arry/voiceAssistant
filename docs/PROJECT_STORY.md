# go42TUM: Voice-Enabled University Application Assistant

## Inspiration

As an international student navigating the complex application process for Technical University of Munich (TUM), I experienced firsthand the overwhelming challenge of sifting through countless official websites, documentation, and requirements. The application process involves visiting multiple portals, reading through extensive FAQs, understanding visa requirements, academic prerequisites, and deadline information scattered across different platforms.

**The pain point was clear**: prospective students spend hours browsing through official websites, often getting lost in bureaucratic language and struggling to find relevant information quickly. This process is particularly challenging for international students who might face language barriers or accessibility issues.

More importantly, I realized that **students with disabilities—especially visually impaired students or those who cannot easily access written information**—face even greater barriers in the application process. Traditional text-heavy websites and PDF documents are often inaccessible, creating significant obstacles for students who rely on screen readers or prefer audio-based interaction.

I realized that a **voice-enabled AI assistant** could revolutionize this experience by providing instant, conversational access to application information, making the process truly accessible, efficient, and user-friendly for everyone, regardless of their abilities or preferred interaction methods.

## What it does

**go42TUM** (pronounced "go-for-TUM") is an intelligent voice assistant specifically designed to help prospective students navigate TUM's application process. The name represents our mission to make it easier for everyone to "go for TUM" - to pursue their educational dreams at Technical University of Munich without barriers. Interestingly, "go42tum" also serves as a placeholder email example (like go42tum@example.com) in our system, symbolizing how we help students take that first step toward their TUM application journey.

The platform provides:

- **Voice-First Interaction**: Natural voice conversations with real-time audio processing and intelligent interruption detection
- **Instant Application Guidance**: Immediate answers about TUM admission requirements, deadlines, documents, and procedures
- **Accessibility-Focused Design**: Screen reader compatibility, keyboard navigation, and high contrast support for visually impaired users
- **Multi-language Support**: Currently supports English and Chinese with seamless language switching
- **Mobile-Optimized Experience**: Progressive Web App (PWA) that works seamlessly on smartphones for on-the-go consultation
- **Session Management**: Persistent chat history and context-aware conversations
- **Smart FAQ System**: Dynamic suggestions and clickable FAQ items for common questions

## How we built it

### Technical Architecture

**Frontend (Next.js 15 + TypeScript)**
- **Progressive Web App (PWA)**: Installable on mobile devices with offline capabilities
- **Voice Integration**: Web Speech API for real-time voice input with volume visualization
- **Real-time Chat**: WebSocket/SSE integration for seamless conversation flow
- **Accessibility-First**: Screen reader compatibility, keyboard navigation, and high contrast support
- **Internationalization**: Multi-language support using i18next

**Backend (FastAPI + Python)**
- **AI-Powered Responses**: Google Gemini 1.5 Flash integration for intelligent conversation
- **Voice Processing**: Real-time audio processing with PCM audio format support using Google Audio Development Kit (ADK)
- **Session Management**: Persistent chat history with PostgreSQL and Redis caching
- **Secure Authentication**: Supabase integration with API key protection
- **Scalable Architecture**: Microservices design ready for production deployment

### Key Features Implemented

1. **Voice-First Interface**
   - Real-time microphone volume visualization
   - Animated voice overlay with pause/resume controls
   - Intelligent voice interruption detection
   - Audio response with natural speech synthesis

2. **Intelligent Chat System**
   - Typewriter effect for engaging user experience
   - Markdown support for rich formatting
   - Auto-scroll with manual override
   - FAQ suggestions when chat is empty

3. **Accessibility & UX**
   - Mobile-responsive design
   - Dark/Light mode with system preference detection
   - Touch-friendly interface optimized for mobile
   - Keyboard navigation and screen reader support

4. **Multi-language Support**
   - Dynamic language switching
   - Persistent language preferences
   - Localized FAQ content

## Challenges we ran into

**Real-time Voice Processing**: Implementing low-latency voice input/output with seamless conversation flow was technically demanding. We solved this by building custom AudioWorklet processors for PCM audio handling and implementing sophisticated buffer management for real-time audio streaming.

**Cross-platform Audio Compatibility**: Web audio APIs behave differently across browsers and mobile devices. We implemented fallback strategies and extensive testing across platforms, using Web Audio API with careful error handling and graceful degradation.

**Accessibility Without Compromise**: Making voice features accessible while maintaining visual appeal required careful balance. We implemented comprehensive keyboard navigation, screen reader compatibility, and high contrast mode while preserving the modern UI using semantic HTML and ARIA labels.

**AI Response Quality & Context**: Ensuring the AI provides accurate, contextual information about TUM applications required fine-tuning prompts for the Gemini model and implementing context-aware session management with robust conversation history.

**Performance Optimization**: Ensuring fast load times and smooth voice interaction on mobile devices was crucial. We implemented code splitting, lazy loading, and optimized audio buffer management using Next.js 15's latest optimization features.

## Accomplishments that we're proud of

- **Voice-First Innovation**: Successfully implemented real-time voice processing with intelligent interruption detection, creating a natural conversation experience that rivals commercial voice assistants
- **Accessibility Leadership**: Built a truly accessible application that works seamlessly for visually impaired users while maintaining modern aesthetics
- **Technical Excellence**: Integrated cutting-edge technologies (Google Gemini AI, Web Audio API, AudioWorklets) into a cohesive, production-ready application
- **Mobile-First Success**: Created a Progressive Web App that installs on mobile devices and works offline, providing university guidance anywhere
- **Cross-Platform Compatibility**: Achieved consistent voice functionality across different browsers and devices despite Web Audio API limitations
- **Multilingual Support**: Implemented dynamic language switching with persistent preferences, making the platform accessible to international students

## What we learned

- **Audio Processing Mastery**: Gained deep understanding of Web Audio API, AudioWorklets, and real-time audio streaming for web applications
- **AI Integration Excellence**: Learned advanced prompt engineering and context management with Google Gemini API for domain-specific use cases
- **Accessibility Engineering**: Practical implementation of WCAG guidelines while maintaining modern UI/UX standards
- **Voice UX Design**: Discovered how voice interfaces require completely different interaction patterns compared to traditional GUIs
- **Progressive Web Apps**: Mastered PWA development with offline capabilities, mobile installation, and service worker management
- **Full-Stack Architecture**: Experience building scalable applications with FastAPI, PostgreSQL, Redis, and Next.js
- **Problem-Solving Approach**: Learning to break down complex user problems into manageable technical solutions with real-world impact

## What's next for TUM VoiceAssistant

**Intelligent Agent System**: We plan to implement a multi-agent architecture to provide comprehensive application support:

- **Report Agent**: An intelligent summarization system that automatically generates personalized reports for users, documenting their questions, the solutions provided, and tracking their application progress. This agent will create detailed application roadmaps and timeline reminders based on each user's specific situation.

- **Email Agent**: When the system encounters questions beyond its knowledge base, this agent will intelligently compose and send emails to relevant TUM staff members, including application offices, international student services, or specific department coordinators. It will format user queries professionally and ensure proper routing to the right departments.

**Enhanced Features Pipeline**:
- **Document Intelligence**: Upload and analyze transcripts, certificates, and application documents with AI-powered feedback
- **Application Timeline**: Personalized deadline tracking with proactive notifications and reminders
- **Multi-University Support**: Expand beyond TUM to support applications for universities across Germany and Europe
- **Peer Connection Network**: Connect prospective students with current students and alumni for mentorship
- **Virtual Campus Integration**: VR/AR campus tours and virtual information sessions
- **Real-time Application Status**: Integration with university portals to track application progress
- **Advanced Analytics**: Detailed insights for TUM admissions offices on common student questions and pain points

---

**go42TUM** represents more than just a technical project—it's a solution born from real student pain points. By combining cutting-edge AI technology with accessibility-first design, we're democratizing access to higher education information and eliminating barriers that prevent students from pursuing their educational dreams.

**Live Demo**: [https://voice-assistant-gilt.vercel.app/](https://voice-assistant-gilt.vercel.app/)

**Tech Stack**: Next.js 15, FastAPI, Google Gemini AI, Google ADK, Supabase, PostgreSQL, Redis, Web Audio API, TypeScript

*Built with ❤️ for the Google Hackathon - Leveraging Google Gemini AI to make education more accessible.* 