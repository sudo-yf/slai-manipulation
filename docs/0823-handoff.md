# 0823 Wrist Handoff

日期：2026-08-23

## 当前进度

- [x] 第一步：腕部遥操作单独测试已完成。
- [x] 策略接口：26DoF 正式数采、Wuji retargeting、UR5e+腕部 8DoF 遥操作、8DoF 数采已分成四个显式组。
- [x] 第二步（软件）：将数采腕部和输出腕部接入 `slai-main` 的统一设备与安全管理。
- [x] 第三步（软件）：接入 UR5e + 腕部 8DoF 联合采集和独立数据契约。
- [ ] 第三步（实物）：采一段包含腕部有效运动的 8D dataset 并检查三路视频。
- [ ] 第四步：完成 dataset 检查和 PI0.5 1-step smoke 训练。

## 2026-08-23 预览低延迟升级

- [x] 三路预览改为 `FFmpeg NVENC H264 -> MediaMTX -> WebRTC/WHEP`。
- [x] 预览使用容量为 1 的最新帧队列；编码器或网页变慢时丢弃旧预览帧，不反压数采。
- [x] MediaMTX 用户服务：`slai-mediamtx.service`，三个流路径为 `primary`、`secondary`、`wrist`。
- [x] `record.leai.me` 已增加三个 WHEP 路由；网页使用 WebRTC，失败时自动回退 JPEG 快照。
- [x] 本机实测三路输入约 `29.9fps`，三路均为 `640x480 H264 Baseline`，无输入 RTP 丢包。
- [x] 局域网直连：网页监听 `0.0.0.0:8765`，WebRTC 协商监听 `0.0.0.0:8889`；同一路由器设备直接打开 `http://192.168.1.102:8765/`，不依赖代理。
- [x] SpaceMouse 常驻监测：独立 UI 服务不再使用 `--camera-only`，以 250Hz 读取并缓存最新状态，SSE 状态刷新间隔降为 50ms。
- [ ] 公网最终验收：需要 Cloudflare Realtime TURN 凭据，或确认路由器已将 UDP/TCP `8189` 转发到本机；Cloudflare Tunnel 本身不能承载 WebRTC UDP 媒体。

### 服务检查

```bash
systemctl --user status slai-mediamtx.service slai-collection-ui.service record-leai-4090.service
curl -s http://127.0.0.1:9997/v3/paths/list | jq '.items[] | {name, ready, tracks, readers}'
curl -s https://record.leai.me/api/status | jq '.cameras[] | {key, fps, age_ms}'
```

策略组可用 `uv run slai-strategies` 只读查看。`ur5e_wrist_8dof_teleop` 只启动 SpaceMouse/UR5e 和 ESP32/OpenRB 腕部，明确关闭 WujiHand 与采集相机。两个运行单元共享停止条件，任一侧退出都会停止另一侧。供应商软链接目录保持只读，运行数据写入主项目 `data/wrist-teleop`。

## 已确认设备

- 数采腕部（ESP32 主手）：`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90160584-if00`
- 输出腕部（OpenRB 从手）：`/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_E261BD20503059384C2E3120FF08262F-if00`
- Espressif JTAG 调试口不是数采数据口，不应作为遥操作输入使用。

供应商 `--check-only` 已返回 `CHECK_OK`，确认两个串口可打开且 OpenRB 运行 V2 `direct_output_cdeg` 固件。腕部遥操作运动测试的完成状态按现场验收结论记录；后续联合验收仍需单独保存 dataset 和训练报告。

## 下一步验收

运行 `scwx --episodes 1` 做实物验收。先验证联合采集中的 wrist state/action 非零且随动作变化，再检查 8 维 state/action、三路视频、时间同步和标定信息，最后用 8DoF schema 运行 PI0.5 1-step 训练并确认 checkpoint 可以重新加载和离线推理。
