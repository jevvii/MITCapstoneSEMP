'use client';

import ProfileManagementDialog from '@/app/components/shared/profile-management-dialog';
import { Badge } from '@/app/components/ui/badge';
import { LazyIcon } from '@/app/components/ui/LazyIcon';
import { Toaster } from '@/app/components/ui/sonner';
import { useAuth } from '@/app/context/AuthContext';
import { openCallSimulationRealtimeStream } from '@/app/lib/assessment/call-simulation-client';
import { getRoleHomePath, navigateToPath } from '@/app/utils/auth-navigation';
import {
    applyUserSettingsToDocument,
    buildDefaultUserSettings,
    loadUserSettings,
    readCachedUserSettings,
    USER_SETTINGS_EVENT,
    type UserDashboardSettings,
} from '@/app/utils/user-settings';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { type ReactNode, useEffect, useMemo, useState } from 'react';

interface SidebarItem {
  label: string;
  icon: ReactNode;
  href: string;
  badge?: number;
  section?: string;
}

const ROLE_LABEL_MAP = {
  trainee: 'Trainee Workspace',
  trainer: 'Trainer Workspace',
  admin: 'Admin Console',
} as const;

const SIDEBAR_BADGE_CACHE_TTL_MS = 30_000;
const sidebarBadgeCache = new Map<string, { expiresAt: number; badges: Record<string, number> }>();
const sidebarBadgeRequests = new Map<string, Promise<Record<string, number> | null>>();

const formatPathLabel = (value: string) =>
  value
    .replace(/\?.*$/, '')
    .replace(/^\[|\]$/g, '')
    .replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());

export function DashboardLayout({
  children,
  sidebarItems,
  userRole = 'trainee',
}: {
  children: React.ReactNode;
  sidebarItems: SidebarItem[];
  userRole?: string;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Removed navSearch and setNavSearch for navigation search removal
  const [dashboardSettings, setDashboardSettings] = useState<UserDashboardSettings>(
    () => readCachedUserSettings() ?? buildDefaultUserSettings(),
  );
  const isSidebarVisible = sidebarOpen || (!dashboardSettings.hideNavigation && !dashboardSettings.topNavigation);
  const [sidebarBadgeMap, setSidebarBadgeMap] = useState<Record<string, number>>({});
  const { user, token, isLoading } = useAuth();
  const pathname = usePathname();
  const resolvedUserRole = userRole === 'admin' || userRole === 'trainer' || userRole === 'trainee'
    ? userRole
    : 'trainee';
  const isTopNavigation = dashboardSettings.topNavigation;
  const isMinifiedSidebar = dashboardSettings.minifyNavigation && !isTopNavigation;
  const isHiddenSidebar = dashboardSettings.hideNavigation && !isTopNavigation;
  const desktopMenuEnabled = isHiddenSidebar;
  const roleHomePath = getRoleHomePath(resolvedUserRole);
  const systemNameLines = ['Speech-Enabled', 'Microlearning Platform'] as const;
  const roleDescriptionMap = {
    trainee: 'Assigned lessons, assessments, call practice, and coaching in one place.',
    trainer: 'Monitor trainee performance, batch health, and coaching follow-through.',
    admin: 'Oversee platform activity, training operations, and system readiness.',
  } as const;
  const roleWorkspaceHintMap = {
    trainee: 'Stay on top of your modules, mock calls, and coaching tasks.',
    trainer: 'Manage learning flow, reviews, and follow-up across your trainees.',
    admin: 'Track system readiness, learning delivery, and organization-wide signals.',
  } as const;
  const contentWidthClass = dashboardSettings.boxedLayout ? 'max-w-[1380px] 2xl:max-w-[1460px]' : 'max-w-none';
  const contentOuterSpacingClass = dashboardSettings.boxedLayout
    ? 'px-2 py-1.5 sm:px-3 sm:py-2 lg:px-3 lg:py-2 xl:px-4 xl:py-2'
    : 'px-2 py-1.5 sm:px-3 sm:py-2 lg:px-3 lg:py-2 xl:px-4 xl:py-2';
  const contentInnerSpacingClass = dashboardSettings.boxedLayout ? 'p-2 sm:p-2.5 lg:p-3 xl:p-3' : 'p-2 sm:p-2.5 lg:p-3 xl:p-3';

  let desktopSidebarStateClass = 'lg:translate-x-0 lg:relative';
  if (isTopNavigation) {
    desktopSidebarStateClass = 'lg:hidden';
  } else if (isHiddenSidebar) {
    desktopSidebarStateClass = sidebarOpen ? 'lg:translate-x-0 lg:fixed' : 'lg:-translate-x-full lg:fixed';
  }

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, []);

  useEffect(() => {
    if (
      typeof window !== 'undefined' &&
      (window.innerWidth < 1024 || dashboardSettings.hideNavigation || dashboardSettings.topNavigation)
    ) {
      setSidebarOpen(false);
    }
  }, [dashboardSettings.hideNavigation, dashboardSettings.topNavigation, pathname]);

  useEffect(() => {
    applyUserSettingsToDocument(dashboardSettings);
  }, [dashboardSettings]);

  useEffect(() => {
    const cachedSettings = readCachedUserSettings();
    if (cachedSettings) {
      setDashboardSettings(cachedSettings);
      applyUserSettingsToDocument(cachedSettings);
    }

    let isMounted = true;
    const handleSettingsUpdate = (event: Event) => {
      const nextSettings = (event as CustomEvent<UserDashboardSettings>).detail;
      if (nextSettings) {
        setDashboardSettings(nextSettings);
      }
    };

    window.addEventListener(USER_SETTINGS_EVENT, handleSettingsUpdate as EventListener);

    void loadUserSettings()
      .then((loadedSettings) => {
        if (isMounted) {
          setDashboardSettings(loadedSettings);
        }
      })
      .catch(() => {
        // Keep cached or default settings when the database read is unavailable.
      });

    return () => {
      isMounted = false;
      window.removeEventListener(USER_SETTINGS_EVENT, handleSettingsUpdate as EventListener);
    };
  }, []);

  useEffect(() => {
    if (isLoading) {
      return;
    }

    if (!user) {
      navigateToPath('/login');
      return;
    }

    if (user.user_role !== resolvedUserRole) {
      navigateToPath(getRoleHomePath(user.user_role));
    }
  }, [isLoading, resolvedUserRole, user]);

  // Avoid eager prefetching of all sidebar routes; Next.js already prefetches visible links when appropriate.

  useEffect(() => {
    if (resolvedUserRole !== 'trainee' || !token) {
      setSidebarBadgeMap({});
      return;
    }

    let isMounted = true;
    const badgeCacheKey = `trainee:${token}`;

    const loadSidebarBadges = async () => {
      const cached = sidebarBadgeCache.get(badgeCacheKey);
      if (cached && cached.expiresAt > Date.now()) {
        if (isMounted) {
          setSidebarBadgeMap(cached.badges);
        }
        return;
      }

      const inFlightRequest = sidebarBadgeRequests.get(badgeCacheKey);
      if (inFlightRequest) {
        const badges = await inFlightRequest;
        if (isMounted && badges) {
          setSidebarBadgeMap(badges);
        }
        return;
      }

      const request = (async () => {
        try {
          const response = await fetch('/api/notifications?limit=20', {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            cache: 'no-store',
          });
          if (!response.ok) {
            return null;
          }
          const payload = (await response.json().catch(() => null)) as
            | { notifications?: Array<{ href?: string | null }> }
            | null;
          const notifications = payload?.notifications || [];
          const certificateBadge = notifications.filter((item) => {
            const href = item.href || '';
            return href.startsWith('/trainee/certificates') || href.startsWith('/trainee/reports?tab=certificates');
          }).length;
          const callSimulationBadge = notifications.filter((item) => (item.href || '').startsWith('/trainee/call-simulation')).length;
          const badges = {
            '/trainee/certificates': certificateBadge,
            '/trainee/call-simulation': callSimulationBadge,
          };
          sidebarBadgeCache.set(badgeCacheKey, {
            expiresAt: Date.now() + SIDEBAR_BADGE_CACHE_TTL_MS,
            badges,
          });
          return badges;
        } catch {
          // Keep the rest of the workspace responsive when notifications are temporarily unavailable.
          return null;
        } finally {
          sidebarBadgeRequests.delete(badgeCacheKey);
        }
      })();

      sidebarBadgeRequests.set(badgeCacheKey, request);

      try {
        if (!isMounted) {
          return;
        }
        const badges = await request;
        if (isMounted && badges) {
          setSidebarBadgeMap(badges);
        }
      } catch {
        // Keep the rest of the workspace responsive when notifications are temporarily unavailable.
      }
    };

    void loadSidebarBadges();
    const intervalId = window.setInterval(() => {
      void loadSidebarBadges();
    }, 60000);

    let stream: EventSource | null = null;
    try {
      stream = openCallSimulationRealtimeStream();
      stream.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as { type?: string };
          if (payload.type === 'session_changed' || payload.type === 'certificate_changed') {
            void loadSidebarBadges();
          }
        } catch {
          // Ignore malformed realtime payloads.
        }
      };
    } catch {
      // Realtime badges are optional; polling keeps this functional.
    }

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
      stream?.close();
    };
  }, [resolvedUserRole, token]);

  const resolvedSidebarItems = sidebarItems.map((item) => ({
    ...item,
    badge: sidebarBadgeMap[item.href] ?? item.badge,
  }));
  // No search: all items shown
  const groupedSidebarItems = useMemo(() => {
    const groups = new Map<string, typeof resolvedSidebarItems>();
    resolvedSidebarItems.forEach((item) => {
      const section = item.section || 'Workspace';
      const items = groups.get(section);
      if (items) {
        items.push(item);
      } else {
        groups.set(section, [item]);
      }
    });
    return Array.from(groups.entries()).map(([section, items]) => ({ section, items }));
  }, [resolvedSidebarItems]);
  const isActivePath = useMemo(
    () => (href: string) => pathname === href || pathname.startsWith(`${href}/`),
    [pathname],
  );
  const activeSidebarItem = useMemo(() => {
    return [...resolvedSidebarItems]
      .sort((left, right) => right.href.length - left.href.length)
      .find((item) => isActivePath(item.href));
  }, [isActivePath, resolvedSidebarItems]);

  const inferredPageLabel = useMemo(() => {
    const segments = pathname.split('/').filter(Boolean);
    const lastSegment = segments.at(-1);
    return lastSegment ? formatPathLabel(lastSegment) : 'Workspace';
  }, [pathname]);
  const currentPageLabel = activeSidebarItem?.label || inferredPageLabel;
  const currentPageSection = activeSidebarItem?.section || ROLE_LABEL_MAP[resolvedUserRole];
  const headerBreadcrumbs = useMemo(() => {
    const items: Array<{ label: string; href?: string }> = [
      { label: ROLE_LABEL_MAP[resolvedUserRole], href: roleHomePath },
    ];

    if (currentPageSection && currentPageSection !== ROLE_LABEL_MAP[resolvedUserRole]) {
      items.push({ label: currentPageSection });
    }

    if (!activeSidebarItem) {
      const rolePrefix = `/${resolvedUserRole}/`;
      const pathAfterRole = pathname.startsWith(rolePrefix)
        ? pathname.slice(rolePrefix.length)
        : pathname.replace(/^\//, '');
      const routeSegments = pathAfterRole.split('/').filter(Boolean);

      routeSegments.slice(0, -1).forEach((segment, index) => {
        const href = `/${resolvedUserRole}/${routeSegments.slice(0, index + 1).join('/')}`;
        items.push({ label: formatPathLabel(segment), href });
      });
    }

    if (currentPageLabel && currentPageLabel !== currentPageSection) {
      items.push({ label: currentPageLabel });
    }

    return items;
  }, [activeSidebarItem, currentPageLabel, currentPageSection, pathname, resolvedUserRole, roleHomePath]);

  const handleSidebarLinkClick = () => {
    if (typeof window !== 'undefined' && (window.innerWidth < 1024 || isHiddenSidebar)) {
      setSidebarOpen(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-foreground px-4 py-6">
        <div className="text-sm text-muted-foreground">Loading dashboard...</div>
      </div>
    );
  }

  if (!user || user.user_role !== resolvedUserRole) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-foreground px-4 py-6">
        <div className="text-sm text-muted-foreground">Redirecting to your workspace...</div>
      </div>
    );
  }

  return (
    <div className="relative flex min-h-0 overflow-x-hidden bg-background text-foreground">
      <a href="#workspace-main-content" className="skip-to-content">
        Skip to main content
      </a>
      <div
        className="absolute inset-0 pointer-events-none bg-gradient-to-b from-[#fbfdff] via-background to-[#eef3f9] dark:from-[#0b1120] dark:via-background dark:to-[#0f172a]"
      />
      <Toaster position="top-right" richColors />
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close navigation menu"
          onClick={() => setSidebarOpen(false)}
          className={`fixed inset-0 z-40 bg-slate-950/55 backdrop-blur-[1px] ${isHiddenSidebar ? '' : 'lg:hidden'}`}
        />
      )}

      {/* Sidebar */}
      <aside
        id="dashboard-sidebar"
        aria-label={`${ROLE_LABEL_MAP[resolvedUserRole]} navigation`}
        aria-hidden={!isSidebarVisible}
        className={`fixed inset-y-0 left-0 z-50 w-[20.5rem] max-w-[94vw] border-r border-sidebar-border bg-sidebar text-sidebar-foreground shadow-[0_24px_60px_-30px_rgba(15,23,42,0.35)] backdrop-blur transition-transform duration-300 transform ${isMinifiedSidebar ? 'lg:w-[6.25rem]' : 'lg:w-[20.5rem]'} lg:max-w-none ${desktopSidebarStateClass} ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        style={{
          backgroundImage:
            'linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)), radial-gradient(circle at top, rgba(59,130,246,0.18), transparent 22%)',
        }}
      >
        <div className="h-full flex flex-col">
          <div className="border-b border-sidebar-border px-3 py-3 sm:px-3.5 sm:py-3.5 lg:px-3.5 lg:py-4">
            <div className={`min-w-0 ${isMinifiedSidebar ? 'lg:flex lg:justify-center' : 'space-y-2.5'}`}>
              <div className={`flex ${isMinifiedSidebar ? 'justify-center' : 'items-center'} gap-2.5`}>
                <div className="relative h-[3.75rem] w-[3.75rem] shrink-0 overflow-hidden rounded-full border border-white/14 bg-white/10 shadow-[0_18px_32px_-24px_rgba(15,23,42,0.95)]">
                  <Image
                    src="/spvlogo.png"
                    alt="St. Peter Velle Technical Training Center logo"
                    fill
                    priority
                    sizes="68px"
                    className="scale-[1.04] rounded-full object-cover"
                  />
                </div>
                <div className={`min-w-0 space-y-0.5 ${isMinifiedSidebar ? 'lg:hidden' : ''}`}>
                  <p className="text-[0.72rem] font-semibold uppercase tracking-[0.16em] text-sidebar-foreground/58">
                    {systemNameLines[0]}
                  </p>
                  <h2 className="text-balance text-[0.95rem] font-semibold leading-5 tracking-normal text-white">
                    {systemNameLines[1]}
                  </h2>
                </div>
              </div>
              {!isMinifiedSidebar ? (
                <div className="rounded-xl border border-white/8 bg-white/[0.05] px-2.5 py-2 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                  <div className="flex items-start justify-between gap-2.5">
                    <div className="min-w-0">
                      <p className="text-[0.72rem] font-semibold uppercase tracking-[0.16em] text-sidebar-foreground/55">
                        {ROLE_LABEL_MAP[resolvedUserRole]}
                      </p>
                      <p className="mt-1 text-sm leading-5 text-sidebar-foreground/68">
                        {roleWorkspaceHintMap[resolvedUserRole]}
                      </p>
                    </div>
                    <div className="shrink-0 rounded-full border border-white/10 bg-white/8 px-2 py-0.5 text-[0.72rem] font-semibold text-sidebar-foreground/70">
                      {resolvedSidebarItems.length} Links
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="flex-1 overflow-y-auto px-2.5 py-3 sm:px-3 sm:py-3.5" aria-label={`${resolvedUserRole} workspace navigation`}>
            {/* Navigation search removed */}
            {groupedSidebarItems.map((group) => (
              <div key={group.section} className="space-y-2 pb-3.5 last:pb-0">
                {!isMinifiedSidebar ? (
                  <div className="flex items-center gap-2 px-2.5">
                    <div className="text-sm font-semibold uppercase tracking-normal text-sidebar-foreground/54">
                      {group.section}
                    </div>
                    <div className="h-px flex-1 bg-white/8" />
                  </div>
                ) : null}
                {group.items.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    prefetch={true}
                    onClick={handleSidebarLinkClick}
                    title={isMinifiedSidebar ? item.label : undefined}
                    aria-label={isMinifiedSidebar ? item.label : undefined}
                    aria-current={isActivePath(item.href) ? 'page' : undefined}
                    className={`group relative flex items-center gap-2.5 rounded-2xl border px-2.5 py-2 transition-[background-color,color,transform,box-shadow,border-color] duration-200 hover:-translate-y-px ${isMinifiedSidebar ? 'lg:justify-center lg:px-2.5' : ''} ${
                      isActivePath(item.href)
                        ? 'border-white/12 bg-white/11 text-white shadow-[0_18px_34px_-24px_rgba(0,0,0,0.55)] ring-1 ring-white/10'
                        : 'border-transparent text-sidebar-foreground/82 hover:border-white/8 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
                    }`}
                  >
                    {isActivePath(item.href) ? (
                      <span className="absolute left-1 top-1/2 h-7 w-1 -translate-y-1/2 rounded-full bg-white/80" />
                    ) : null}
                    <span
                      className={`inline-flex size-9 shrink-0 items-center justify-center rounded-xl border transition ${
                        isActivePath(item.href)
                          ? 'border-white/12 bg-white/10 text-white'
                          : 'border-white/6 bg-white/[0.04] text-sidebar-foreground/80 group-hover:border-white/10 group-hover:bg-white/[0.08] group-hover:text-sidebar-accent-foreground'
                      }`}
                    >
                      {item.icon}
                    </span>
                    <div className={`min-w-0 flex-1 ${isMinifiedSidebar ? 'lg:hidden' : ''}`}>
                      <div className="line-clamp-2 text-[0.9rem] font-medium leading-5 text-pretty">{item.label}</div>
                    </div>
                    {item.badge && !isMinifiedSidebar ? (
                      <Badge variant="danger" className="min-w-5 justify-center px-2 py-0.5 text-[0.68rem]">
                        {item.badge}
                      </Badge>
                    ) : null}
                    {isActivePath(item.href) && !item.badge && !isMinifiedSidebar ? (
                      <span className="size-2 shrink-0 rounded-full bg-white/80" />
                    ) : null}
                  </Link>
                ))}
              </div>
            ))}
            {/* No empty state needed since search is removed; all items always shown */}
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        {/* Top Navigation Bar */}
        <header
          role="banner"
          className={`workspace-topbar flex flex-col gap-2 border-b border-border/80 px-2.5 py-2 shadow-sm sm:px-3 sm:py-2.5 lg:flex-row lg:items-center lg:justify-between lg:px-3.5 xl:px-4 ${
            dashboardSettings.fixedHeader ? 'sticky top-0 z-30' : ''
          }`}
        >
          <div className="flex min-w-0 flex-1 items-start gap-2.5 sm:items-center sm:gap-3">
            <button
              type="button"
              onClick={() => setSidebarOpen((prev) => !prev)}
              aria-controls="dashboard-sidebar"
              aria-expanded={sidebarOpen}
              className={`inline-flex items-center justify-center rounded-2xl border border-border bg-background p-2.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground ${
                desktopMenuEnabled ? 'lg:inline-flex' : 'lg:hidden'
              }`}
              aria-label={sidebarOpen ? 'Close navigation menu' : 'Open navigation menu'}
              title={sidebarOpen ? 'Close menu' : 'Open menu'}
            >
              {sidebarOpen ? <LazyIcon name="X" size={22} /> : <LazyIcon name="Menu" size={22} />}
            </button>

            <button type="button" onClick={() => {}} aria-hidden className="sr-only" />

            <div className="min-w-0 space-y-1.5">
              <div className="min-w-0 space-y-1">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="rounded-full border border-primary/12 bg-primary/6 px-2 py-0.5 text-[0.72rem] font-semibold uppercase tracking-normal text-primary">
                    {ROLE_LABEL_MAP[resolvedUserRole]}
                  </span>
                  <span className="rounded-full border border-border/70 bg-background px-2 py-0.5 text-[0.72rem] font-semibold uppercase tracking-normal text-muted-foreground">
                    {currentPageSection}
                  </span>
                  <Badge variant="info" className="px-2 py-0.5 text-[0.72rem]">
                    Live Workspace
                  </Badge>
                </div>
                <div className="workspace-breadcrumbs">
                  {headerBreadcrumbs.map((item, index) => (
                    <div key={`${item.label}-${index}`} className="contents">
                      {index > 0 ? (
                        <span className="workspace-breadcrumb-separator">
                          <LazyIcon name="ChevronRight" size={14} />
                        </span>
                      ) : null}
                      {item.href ? (
                        <Link href={item.href} className="transition-colors hover:text-foreground">
                          {item.label}
                        </Link>
                      ) : (
                        <span>{item.label}</span>
                      )}
                    </div>
                  ))}
                </div>
                <h1 className="line-clamp-2 text-balance text-[1.05rem] font-bold tracking-normal text-foreground sm:text-[1.2rem] lg:text-[1.38rem] xl:text-[1.5rem]">
                  {currentPageLabel}
                </h1>
                <p className="hidden max-w-3xl text-sm leading-5 text-muted-foreground sm:block">
                  {roleDescriptionMap[resolvedUserRole]}
                </p>
                <p className="text-sm leading-5 text-muted-foreground/85 sm:hidden">
                  {roleWorkspaceHintMap[resolvedUserRole]}
                </p>
              </div>
            </div>
          </div>

          <div className="flex w-full flex-col gap-2.5 sm:flex-row sm:items-center sm:gap-2.5 lg:w-auto lg:justify-end">
            <ProfileManagementDialog />
          </div>
        </header>

        {isTopNavigation ? (
          <div className="hidden border-b border-border bg-background/88 backdrop-blur lg:block">
            <div className={`mx-auto w-full px-4 py-3 sm:px-5 lg:px-6 ${contentWidthClass}`}>
              <nav aria-label={`${resolvedUserRole} workspace sections`} className="dashboard-balanced-grid">
                {groupedSidebarItems.map((group) => (
                  <div
                    key={`top-group-${group.section}`}
                    className="rounded-xl border border-border/70 bg-white/90 px-3 py-2.5 shadow-[0_16px_34px_-30px_rgba(15,23,42,0.26)]"
                  >
                    <div className="mb-2.5 flex items-center gap-2">
                      <div className="text-[0.68rem] font-semibold uppercase tracking-normal text-muted-foreground">
                        {group.section}
                      </div>
                      <div className="h-px flex-1 bg-border/80" />
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {group.items.map((item) => (
                          <Link
                          key={`top-${item.href}`}
                          href={item.href}
                          aria-current={isActivePath(item.href) ? 'page' : undefined}
                          className={`inline-flex min-h-9 items-center gap-2 rounded-full border px-2.5 py-1.5 text-[0.84rem] font-medium transition-[background-color,color,border-color,box-shadow] ${
                            isActivePath(item.href)
                              ? 'border-primary/16 bg-primary text-primary-foreground shadow-[0_14px_28px_-20px_rgba(29,86,216,0.45)]'
                              : 'border-border/80 bg-background text-muted-foreground hover:border-primary/18 hover:bg-secondary hover:text-foreground'
                          }`}
                        >
                          <span>{item.icon}</span>
                          <span>{item.label}</span>
                          {item.badge ? (
                            <span
                              className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                isActivePath(item.href)
                                  ? 'bg-white/18 text-white'
                                  : 'bg-red-100 text-red-800'
                              }`}
                            >
                              {item.badge}
                            </span>
                          ) : null}
                        </Link>
                      ))}
                    </div>
                  </div>
                ))}
              </nav>
            </div>
          </div>
        ) : null}

        {/* Page Content */}
        <main
          id="workspace-main-content"
          role="main"
          className="flex-1 min-h-0 overflow-x-hidden overflow-y-auto overscroll-contain [scrollbar-gutter:stable_both-edges]"
          tabIndex={-1}
        >
          <div className={`dashboard-page mx-auto w-full ${contentOuterSpacingClass} ${contentWidthClass}`}>
            <div
              className={`workspace-shell rounded-[1.35rem] text-card-foreground ${contentInnerSpacingClass}`}
            >
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
