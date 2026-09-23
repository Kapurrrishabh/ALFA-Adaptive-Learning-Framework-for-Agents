import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next writes AGENTS.md and CLAUDE.md into this directory on every run. The repository already
  // carries its own instructions, and a generated file that quietly overrides them is worse than none.
  agentRules: false,
};

export default nextConfig;
