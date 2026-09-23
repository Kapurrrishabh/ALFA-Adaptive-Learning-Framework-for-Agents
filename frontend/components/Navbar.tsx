"use client";

import { useRouter, usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, LogOut, Menu, X } from "lucide-react";
import toast from "react-hot-toast";

import { useAuth } from "@/hooks/useAuth";

// Only the two the backend can serve. A link to a page whose endpoints do not exist is worse than no link.
const NAV_LINKS = [
  { name: "Ask", path: "/" },
  { name: "Portfolio", path: "/portfolio" },
];

function ProfileDropdown({ open, email, onSignOut }: { open: boolean; email: string; onSignOut: () => void }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -8, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.97 }}
          className="absolute right-0 mt-2 w-60 rounded-2xl bg-[#0c0c0c]/95 backdrop-blur-2xl border border-white/[0.09] shadow-[0_12px_50px_rgba(0,0,0,0.6)] overflow-hidden"
        >
          <div className="px-4 py-3 border-b border-white/[0.06]">
            <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Signed in as</div>
            <div className="text-sm text-gray-200 font-semibold truncate">{email}</div>
          </div>
          <button
            onClick={onSignOut}
            className="w-full flex items-center gap-2.5 px-4 py-3 text-sm text-gray-300 hover:bg-white/[0.05] hover:text-white transition-colors"
          >
            <LogOut size={15} /> Sign out
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function MobileMenu({
  open, onClose, email, pathname, onSignOut,
}: {
  open: boolean; onClose: () => void; email: string | null; pathname: string; onSignOut: () => void;
}) {
  const router = useRouter();
  const navigate = (path: string) => { router.push(path); onClose(); };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[950] bg-[#050505]/95 backdrop-blur-xl md:hidden p-6"
        >
          <div className="flex items-center justify-between mb-10">
            <span className="text-xl font-black tracking-tight">FinIntel</span>
            <button onClick={onClose} className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.07] text-gray-400">
              <X size={20} />
            </button>
          </div>
          <div className="space-y-2">
            {NAV_LINKS.map((link) => (
              <button
                key={link.path}
                onClick={() => navigate(link.path)}
                className={`w-full text-left px-4 py-3.5 rounded-xl text-base font-semibold transition-colors ${
                  pathname === link.path ? "bg-white/[0.09] text-white" : "text-gray-500 hover:text-gray-200"
                }`}
              >
                {link.name}
              </button>
            ))}
          </div>
          {email ? (
            <button
              onClick={() => { onSignOut(); onClose(); }}
              className="mt-10 w-full flex items-center justify-center gap-2 py-3.5 rounded-xl bg-white/[0.04] border border-white/[0.07] text-sm font-semibold text-gray-300"
            >
              <LogOut size={15} /> Sign out of {email}
            </button>
          ) : (
            <button
              onClick={() => navigate("/login")}
              className="mt-10 w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 text-sm font-bold text-white"
            >
              Access Terminal
            </button>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default function Navbar() {
  const router = useRouter();
  const pathname = usePathname();
  const { user, loading, signOut } = useAuth();
  const [isScrolled, setIsScrolled] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onScroll = () => setIsScrolled(window.scrollY > 10);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) setProfileOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const handleSignOut = useCallback(async () => {
    await signOut();
    toast.success("Signed out");
    setProfileOpen(false);
    router.push("/login");
  }, [signOut, router]);

  return (
    <>
      <header className={`fixed inset-x-0 z-[900] flex justify-center px-4 md:px-6 transition-all duration-500 ${isScrolled ? "top-3" : "top-5"}`}>
        <div className={`w-full max-w-7xl flex items-center justify-between transition-all duration-300 rounded-2xl ${
          isScrolled
            ? "bg-[#080808]/85 backdrop-blur-2xl border border-white/[0.09] px-5 py-3 shadow-[0_12px_50px_rgba(0,0,0,0.6)]"
            : "bg-transparent border border-transparent px-3 py-4"
        }`}>
          <motion.button
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            onClick={() => router.push("/")}
            className="flex items-center gap-2.5 group focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-xl"
          >
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-[0_0_14px_rgba(59,130,246,0.4)] group-hover:shadow-[0_0_22px_rgba(59,130,246,0.6)] transition-shadow">
              <Activity size={17} className="text-white" />
            </div>
            <span className="text-xl font-black text-white tracking-tight">FinIntel</span>
          </motion.button>

          <nav className="hidden md:flex items-center gap-0.5 p-1 rounded-full bg-white/[0.03] border border-white/[0.06]">
            {NAV_LINKS.map((link) => {
              const active = pathname === link.path;
              return (
                <button
                  key={link.path}
                  onClick={() => router.push(link.path)}
                  className={`relative px-4 py-2 rounded-full text-sm font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${active ? "text-white" : "text-gray-500 hover:text-gray-200"}`}
                >
                  {active && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 bg-white/[0.09] rounded-full"
                      transition={{ type: "spring", stiffness: 350, damping: 32 }}
                    />
                  )}
                  <span className="relative z-[901]">{link.name}</span>
                </button>
              );
            })}
          </nav>

          <div className="hidden md:flex items-center gap-2">
            {loading ? (
              <div className="w-32 h-9 rounded-full bg-white/[0.06] animate-pulse" />
            ) : user ? (
              <div className="relative" ref={profileRef}>
                <motion.button
                  whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
                  onClick={() => setProfileOpen((was) => !was)}
                  className={`flex items-center gap-2.5 pl-3 pr-3 py-2 rounded-full border transition-all focus-visible:ring-2 focus-visible:ring-blue-500 outline-none ${profileOpen ? "bg-white/[0.08] border-white/20" : "bg-white/[0.04] border-white/[0.08] hover:bg-white/[0.07] hover:border-white/15"}`}
                >
                  <span className="text-sm text-gray-200 font-semibold max-w-[120px] truncate">
                    {user.email.split("@")[0]}
                  </span>
                </motion.button>
                <ProfileDropdown open={profileOpen} email={user.email} onSignOut={handleSignOut} />
              </div>
            ) : (
              <motion.button
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
                onClick={() => router.push("/login")}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 text-sm font-bold text-white shadow-[0_0_14px_rgba(59,130,246,0.3)] hover:shadow-[0_0_24px_rgba(59,130,246,0.5)] transition-all"
              >
                Access Terminal
              </motion.button>
            )}
          </div>

          <button
            onClick={() => setMobileMenuOpen(true)}
            className="md:hidden p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.07] text-gray-400 hover:text-white transition-colors focus-visible:ring-2 focus-visible:ring-blue-500 outline-none"
          >
            <Menu size={20} />
          </button>
        </div>
      </header>

      <MobileMenu
        open={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        email={user?.email ?? null}
        pathname={pathname}
        onSignOut={handleSignOut}
      />
    </>
  );
}
