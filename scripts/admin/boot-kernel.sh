#!/usr/bin/env bash
# Root-only, run by hand. Swaps the Ubuntu Rockchip kernel installed on the NVMe root
# into the Orange Pi's U-Boot boot partition (boot.scr loads Image, uInitrd, dtb/<fdtfile>).
#
# usage: boot-kernel.sh install BOOT_MOUNTPOINT KERNEL_RELEASE   e.g. ... /mnt/sdcard/boot 6.1.0-1026-rockchip
#        boot-kernel.sh restore BOOT_MOUNTPOINT                  put the saved vendor files back
set -euo pipefail

ACTION=$1
BOOT=$2
FDT=$(sed -n 's/^fdtfile=//p' "$BOOT/orangepiEnv.txt")

mount -o remount,rw "$BOOT"
case "$ACTION" in
  install)
    KVER=$3
    for f in Image uInitrd "dtb/$FDT"; do
      [ -e "$BOOT/$f.vendor" ] || cp -a "$BOOT/$f" "$BOOT/$f.vendor"
    done
    zcat -f "/boot/vmlinuz-$KVER" > "$BOOT/Image"
    mkimage -A arm64 -O linux -T ramdisk -C none -n uInitrd -d "/boot/initrd.img-$KVER" "$BOOT/uInitrd"
    cp "/lib/firmware/$KVER/device-tree/$FDT" "$BOOT/dtb/$FDT"
    ;;
  restore)
    for f in Image uInitrd "dtb/$FDT"; do
      cp -a "$BOOT/$f.vendor" "$BOOT/$f"
    done
    ;;
esac
sync
mount -o remount,ro "$BOOT"
