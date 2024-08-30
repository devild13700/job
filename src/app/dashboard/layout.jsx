import Link from 'next/link'
import {
  Home,
  LineChart,
  Package,
  Package2,
  PanelLeft,
  Settings,
  ShoppingCart,
  Users2,
} from 'lucide-react'

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '../../components/ui/breadcrumb'
import { Button } from '../../components/ui/button'
import { Sheet, SheetContent, SheetTrigger } from '../../components/ui/sheet'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '../../components/ui/tooltip'
import { Analytics } from '@vercel/analytics/react'
import { User } from './user'
import { VercelLogo } from '../../components/icons'
import Providers from './providers'
import { NavItem } from './nav-item'
import { SearchInput } from './search'

export default function DashboardLayout({ children }) {
  return (
    <Providers>
      <main className='flex min-h-screen  flex-col bg-muted/40'>
        <DesktopNav />
        <div className='flex flex-col sm:gap-4 sm:py-4 sm:pl-14'>
          <header className='sticky top-0 z-30 flex h-14 items-center gap-4 border-b bg-background px-4 sm:static sm:h-auto sm:border-0 sm:bg-transparent sm:px-6'>
            <MobileNav />
            <SearchInput />
            <User />
          </header>
          <main className='grid flex-1 items-start gap-2 p-4 sm:py-0 md:gap-4 bg-muted/40'>
            {children}
          </main>
        </div>
        <Analytics />
      </main>
    </Providers>
  )
}

function DesktopNav() {
  return (
    <aside className='fixed inset-y-0 left-0 z-10 hidden w-14 flex-col border-r bg-background sm:flex'>
      <nav className='flex flex-col items-center gap-4 px-2 sm:py-5'>
        <Link
          href='/customers'
          className='group flex h-9 w-9 shrink-0 items-center justify-center gap-2 rounded-full bg-primary text-lg font-semibold text-primary-foreground md:h-8 md:w-8 md:text-base'
        >
          <VercelLogo className='h-3 w-3 transition-all group-hover:scale-110' />
          <span className='sr-only'>Acme Inc</span>
        </Link>

        <NavItem href='/dashboard' label='Dashboard'>
          <Home className='h-5 w-5' />
        </NavItem>

        <NavItem href='/dashboard/recruiters' label='Recruiters'>
          <Users2 className='h-5 w-5' />
        </NavItem>

        <NavItem href='/dashboard/candidates' label='Candidates'>
          <Users2 className='h-5 w-5' />
        </NavItem>
      </nav>
    </aside>
  )
}

function MobileNav() {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button size='icon' variant='outline' className='sm:hidden'>
          <PanelLeft className='h-5 w-5' />
          <span className='sr-only'>Toggle Menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side='left' className='sm:max-w-xs'>
        <nav className='grid gap-6 text-lg font-medium'>
          <Link
            href='#'
            className='group flex h-10 w-10 shrink-0 items-center justify-center gap-2 rounded-full bg-primary text-lg font-semibold text-primary-foreground md:text-base'
          >
            <Package2 className='h-5 w-5 transition-all group-hover:scale-110' />
            <span className='sr-only'></span>
          </Link>
          <Link
            href='/dashboard'
            className='flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground'
          >
            <Home className='h-5 w-5' />
            Dashboard
          </Link>
          <Link
            href='/dashboard/recruiters'
            className='flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground'
          >
            <Users2 className='h-5 w-5' />
            Recruiters
          </Link>
          <Link
            href='/dashboard/candidates'
            className='flex items-center gap-4 px-2.5 text-foreground'
          >
            <Users2 className='h-5 w-5' />
            Candidates
          </Link>
        </nav>
      </SheetContent>
    </Sheet>
  )
}