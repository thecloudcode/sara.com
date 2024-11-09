"use client";

import Link from "next/link";
import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { HoveredLink, Menu, MenuItem, ProductItem } from "@/components/ui/navbar-menu";

export default function Navbar({ className }: { className?: string }) {
  const [active, setActive] = useState<string | null>(null);

  return (
    <div className={cn("fixed top-10 inset-x-0 max-w-2xl mx-auto z-50", className)}>
      <Menu setActive={setActive}>
        <Link href="/">
        <p className="cursor-pointer text-black hover:opacity-[0.9] dark:text-white">
            Home</p>
        </Link>
        <MenuItem setActive={setActive} active={active} item="Products">
          <div className="text-sm grid grid-cols-2 gap-10 p-4">
            <ProductItem
              title="Sneakers"
              href="/sneakers"
              src="https://i.pinimg.com/564x/1f/9d/ca/1f9dcac4e16c4b2582b43f70cfb0f630.jpg"
              description="Step up your style game with the latest in trendy, comfortable sneakers designed for every occasion."
            />
            <ProductItem
              title="Smartwatches & Wearables"
              href="https://tailwindmasterkit.com"
              src="https://i.pinimg.com/564x/cd/cd/ab/cdcdabfb9b5259236eb5b6a7d4f17666.jpg"
              description="Stay connected and track your health with cutting-edge smartwatches and wearables built for active lifestyles."
            />
          </div>
        </MenuItem>
        <MenuItem setActive={setActive} active={active} item="Pricing">
          <div className="flex flex-col space-y-4 text-sm">
            <HoveredLink href="/hobby">Hobby</HoveredLink>
            <HoveredLink href="/individual">Individual</HoveredLink>
            <HoveredLink href="/team">Team</HoveredLink>
            <HoveredLink href="/enterprise">Enterprise</HoveredLink>
          </div>
        </MenuItem>
      </Menu>
    </div>
  );
}
