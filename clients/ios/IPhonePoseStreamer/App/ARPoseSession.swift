import ARKit
import Combine
import Foundation
import simd
import UIKit

final class ARPoseSession: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published private(set) var trackingStatus = "正在初始化"
    @Published private(set) var transportStatus = "正在启动"
    @Published private(set) var position = SIMD3<Float>(repeating: 0)
    @Published private(set) var framesPerSecond = 0

    private let server = PoseTCPServer()
    private var sequence: UInt64 = 0
    private var lastUIUpdate = 0.0
    private var frameCount = 0
    private var rateWindowStart = ProcessInfo.processInfo.systemUptime

    override init() {
        super.init()
        session.delegate = self
        server.statusChanged = { [weak self] value in
            self?.transportStatus = value
        }
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            trackingStatus = "此设备不支持 ARKit 世界跟踪"
            return
        }
        UIApplication.shared.isIdleTimerDisabled = true
        server.start()
        run(reset: true)
    }

    func pause() {
        session.pause()
        server.stop()
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func recenter() {
        run(reset: true)
    }

    private func run(reset: Bool) {
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        session.run(configuration, options: options)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let tracking = trackingLabel(frame.camera.trackingState)
        let packet = PosePacket(
            sequence: sequence,
            timestampSeconds: frame.timestamp,
            tracking: tracking.machine,
            transform: frame.camera.transform
        )
        sequence &+= 1
        server.send(packet)

        frameCount += 1
        let now = ProcessInfo.processInfo.systemUptime
        let elapsed = now - rateWindowStart
        if elapsed >= 1.0 {
            let measuredRate = Int((Double(frameCount) / elapsed).rounded())
            frameCount = 0
            rateWindowStart = now
            DispatchQueue.main.async { [weak self] in
                self?.framesPerSecond = measuredRate
            }
        }
        guard now - lastUIUpdate >= 0.1 else { return }
        lastUIUpdate = now
        let transform = frame.camera.transform
        let measuredPosition = SIMD3<Float>(
            transform.columns.3.x,
            transform.columns.3.y,
            transform.columns.3.z
        )
        DispatchQueue.main.async { [weak self] in
            self?.position = measuredPosition
            self?.trackingStatus = tracking.display
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { [weak self] in
            self?.trackingStatus = "AR 会话错误: \(error.localizedDescription)"
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in
            self?.trackingStatus = "AR 会话已中断"
        }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        DispatchQueue.main.async { [weak self] in
            self?.run(reset: true)
        }
    }

    private func trackingLabel(_ state: ARCamera.TrackingState) -> (machine: String, display: String) {
        switch state {
        case .normal:
            return ("normal", "跟踪正常")
        case .notAvailable:
            return ("not_available", "跟踪不可用")
        case .limited(let reason):
            switch reason {
            case .initializing:
                return ("limited_initializing", "正在初始化")
            case .excessiveMotion:
                return ("limited_excessive_motion", "移动过快")
            case .insufficientFeatures:
                return ("limited_insufficient_features", "环境纹理不足")
            case .relocalizing:
                return ("limited_relocalizing", "正在重新定位")
            @unknown default:
                return ("limited_unknown", "跟踪受限")
            }
        }
    }
}
