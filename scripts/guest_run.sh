#!/bin/bash
set -e

CONFIG_FILE="/home/rog/lpu-workspace/scale/driver/pciemu/release/config"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found! Is QEMU running?" >&2
    exit 1
fi

PORT=$(grep -w 'qemu_vm_ssh_port' "$CONFIG_FILE" | awk -F ':' '{print $2}' | tr -d ' \r\n')
if [ -z "$PORT" ]; then
    echo "Error: Could not extract qemu_vm_ssh_port from $CONFIG_FILE" >&2
    exit 1
fi

# Run command inside guest VM
sshpass -p Linux ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=60 -p "$PORT" root@127.0.0.1 "$@"
