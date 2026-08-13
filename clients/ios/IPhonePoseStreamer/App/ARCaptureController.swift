import ARKit
import Foundation
import simd

final class ARCaptureController: NSObject, ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var trackingDescription = "Ready"
    @Published private(set) var position = SIMD3<Float>(repeating: 0)
    @Published private(set) var orientationDegrees = SIMD3<Float>(repeating: 0)
    @Published private(set) var frameRate = 0
    @Published private(set) var depthAvailable = false
    @Published private(set) var latestTimestamp = 0.0
    @Published private(set) var transportDescription = "Stream stopped"
    @Published private(set) var sentFrameCount = 0
    @Published private(set) var supersededFrameCount = 0
    @Published private(set) var cloudDescription = "Cloud not configured"
    @Published private(set) var cloudSentFrameCount = 0
    @Published private(set) var cloudSupersededFrameCount = 0
    @Published private(set) var teleoperationEnabled = false
    @Published var cloudEndpoint: String
    @Published var cloudSessionID: String
    @Published var cloudToken: String
    @Published var cloudViewerURL: String

    private weak var session: ARSession?
    private let transport = PoseTCPServer()
    private let cloudPublisher = PoseWebSocketPublisher()
    private var sequence: UInt64 = 0
    private var frameCount = 0
    private var rateWindowStart = CACurrentMediaTime()
    private let teleoperationLock = NSLock()
    private var teleoperationEnabledValue = false
    private var teleoperationEpoch: UInt64 = 0

    override init() {
        let defaults = UserDefaults.standard
        let launchOptions = Self.launchOptions()
        cloudEndpoint = launchOptions["pose-hub-endpoint"] ?? defaults.string(forKey: "cloudEndpoint") ?? "wss://6d.leai.me"
        cloudSessionID = launchOptions["pose-hub-session"] ?? defaults.string(forKey: "cloudSessionID") ?? ""
        cloudToken = launchOptions["pose-hub-token"] ?? defaults.string(forKey: "cloudToken") ?? ""
        cloudViewerURL = launchOptions["pose-hub-viewer-url"] ?? defaults.string(forKey: "cloudViewerURL") ?? ""
        super.init()
        if !launchOptions.isEmpty {
            defaults.set(cloudEndpoint, forKey: "cloudEndpoint")
            defaults.set(cloudSessionID, forKey: "cloudSessionID")
            defaults.set(cloudToken, forKey: "cloudToken")
            defaults.set(cloudViewerURL, forKey: "cloudViewerURL")
        }
        transport.statusChanged = { [weak self] status in
            self?.transportDescription = status
        }
        transport.countersChanged = { [weak self] sent, superseded in
            self?.sentFrameCount = sent
            self?.supersededFrameCount = superseded
        }
        cloudPublisher.statusChanged = { [weak self] status in
            self?.cloudDescription = status
        }
        cloudPublisher.countersChanged = { [weak self] sent, superseded in
            self?.cloudSentFrameCount = sent
            self?.cloudSupersededFrameCount = superseded
        }
    }

    func attach(to session: ARSession) {
        self.session = session
        session.delegate = self
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            updateOnMain { self.trackingDescription = "World tracking is unavailable on this device" }
            return
        }

        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        if let lowLatencyFormat = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ $0.framesPerSecond >= 60 })
            .min(by: {
                $0.imageResolution.width * $0.imageResolution.height <
                    $1.imageResolution.width * $1.imageResolution.height
            }) {
            configuration.videoFormat = lowLatencyFormat
        }
        updateOnMain { self.depthAvailable = false }

        transport.start()
        startCloudPublisher()
        session?.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        frameCount = 0
        rateWindowStart = CACurrentMediaTime()
        updateOnMain {
            self.isRunning = true
            self.trackingDescription = "Initializing"
        }
    }

    func stop() {
        setTeleoperationEnabled(false)
        session?.pause()
        transport.stop()
        cloudPublisher.stop()
        updateOnMain {
            self.isRunning = false
            self.trackingDescription = "Paused"
        }
    }

    func recenter() {
        guard isRunning else {
            start()
            return
        }
        start()
    }

    func setTeleoperationEnabled(_ enabled: Bool) {
        guard isRunning || !enabled else { return }
        teleoperationLock.lock()
        let changed = teleoperationEnabledValue != enabled
        if changed && enabled {
            teleoperationEpoch &+= 1
        }
        teleoperationEnabledValue = enabled
        let epoch = teleoperationEpoch
        teleoperationLock.unlock()
        guard changed else { return }
        updateOnMain {
            self.teleoperationEnabled = enabled
            if enabled {
                self.trackingDescription = "Teleoperation enabled (reference \(epoch))"
            }
        }
    }

    private func teleoperationState() -> (enabled: Bool, epoch: UInt64) {
        teleoperationLock.lock()
        defer { teleoperationLock.unlock() }
        return (teleoperationEnabledValue, teleoperationEpoch)
    }

    func posePayload() -> String {
        let position = self.position
        let rotation = self.orientationDegrees
        return String(
            format: "{\"timestamp\":%.3f,\"position_m\":[%.4f,%.4f,%.4f],\"rotation_deg\":[%.2f,%.2f,%.2f],\"depthAvailable\":%@}",
            latestTimestamp,
            position.x, position.y, position.z,
            rotation.x, rotation.y, rotation.z,
            depthAvailable ? "true" : "false"
        )
    }

    func saveCloudSettings() {
        let defaults = UserDefaults.standard
        defaults.set(cloudEndpoint, forKey: "cloudEndpoint")
        defaults.set(cloudSessionID, forKey: "cloudSessionID")
        defaults.set(cloudToken, forKey: "cloudToken")
        defaults.set(cloudViewerURL, forKey: "cloudViewerURL")
        if isRunning {
            startCloudPublisher()
        } else {
            cloudDescription = hasCloudConfiguration ? "Cloud ready" : "Cloud not configured"
        }
    }

    private var hasCloudConfiguration: Bool {
        !cloudEndpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !cloudSessionID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !cloudToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func startCloudPublisher() {
        guard hasCloudConfiguration else {
            cloudPublisher.stop()
            cloudDescription = "Cloud not configured"
            return
        }
        cloudPublisher.start(endpoint: cloudEndpoint, sessionID: cloudSessionID, token: cloudToken)
    }

    func openViewer() {
        guard let url = URL(string: cloudViewerURL) else { return }
        UIApplication.shared.open(url)
    }

    private func updateOnMain(_ update: @escaping () -> Void) {
        if Thread.isMainThread {
            update()
        } else {
            DispatchQueue.main.async(execute: update)
        }
    }

    private static func launchOptions() -> [String: String] {
        let arguments = ProcessInfo.processInfo.arguments
        var values: [String: String] = [:]
        for (index, argument) in arguments.enumerated() where argument.hasPrefix("--") && index + 1 < arguments.count {
            let key = String(argument.dropFirst(2))
            let value = arguments[index + 1]
            if !value.hasPrefix("--") {
                values[key] = value
            }
        }
        return values
    }
}

extension ARCaptureController: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let transform = frame.camera.transform
        let translation = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        let euler = simd_quatf(transform).eulerAngles * (180 / .pi)
        let state = description(for: frame.camera.trackingState)
        let timestamp = frame.timestamp
        let teleoperation = teleoperationState()
        transport.send(
            PosePacket(
                sequence: sequence,
                timestampSeconds: timestamp,
                tracking: machineDescription(for: frame.camera.trackingState),
                teleoperationEnabled: teleoperation.enabled,
                teleoperationEpoch: teleoperation.epoch,
                transform: transform
            )
        )
        cloudPublisher.send(
            PosePacket(
                sequence: sequence,
                timestampSeconds: timestamp,
                tracking: machineDescription(for: frame.camera.trackingState),
                teleoperationEnabled: teleoperation.enabled,
                teleoperationEpoch: teleoperation.epoch,
                transform: transform
            )
        )
        sequence &+= 1

        frameCount += 1
        let now = CACurrentMediaTime()
        let elapsed = now - rateWindowStart
        let currentRate = elapsed >= 1 ? Int((Double(frameCount) / elapsed).rounded()) : frameRate
        if elapsed >= 1 {
            frameCount = 0
            rateWindowStart = now
        }

        updateOnMain {
            self.position = translation
            self.orientationDegrees = euler
            self.trackingDescription = state
            self.latestTimestamp = timestamp
            self.frameRate = currentRate
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        updateOnMain {
            self.trackingDescription = "Session error: \(error.localizedDescription)"
            self.isRunning = false
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        updateOnMain { self.trackingDescription = "Session interrupted" }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        updateOnMain { self.trackingDescription = "Session interruption ended. Restart tracking." }
    }

    private func description(for state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "Tracking normal"
        case .notAvailable:
            return "Tracking unavailable"
        case .limited(let reason):
            switch reason {
            case .initializing: return "Tracking limited: initializing"
            case .excessiveMotion: return "Tracking limited: move slower"
            case .insufficientFeatures: return "Tracking limited: point at a textured surface"
            case .relocalizing: return "Tracking limited: relocalizing"
            @unknown default: return "Tracking limited"
            }
        }
    }

    private func machineDescription(for state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "normal"
        case .notAvailable:
            return "not_available"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited_initializing"
            case .excessiveMotion: return "limited_excessive_motion"
            case .insufficientFeatures: return "limited_insufficient_features"
            case .relocalizing: return "limited_relocalizing"
            @unknown default: return "limited_unknown"
            }
        }
    }
}

private extension simd_quatf {
    var eulerAngles: SIMD3<Float> {
        let q = normalized
        let sinRollCosPitch = 2 * (q.real * q.imag.x + q.imag.y * q.imag.z)
        let cosRollCosPitch = 1 - 2 * (q.imag.x * q.imag.x + q.imag.y * q.imag.y)
        let roll = atan2(sinRollCosPitch, cosRollCosPitch)

        let sinPitch = 2 * (q.real * q.imag.y - q.imag.z * q.imag.x)
        let pitch = abs(sinPitch) >= 1 ? copysign(.pi / 2, sinPitch) : asin(sinPitch)

        let sinYawCosPitch = 2 * (q.real * q.imag.z + q.imag.x * q.imag.y)
        let cosYawCosPitch = 1 - 2 * (q.imag.y * q.imag.y + q.imag.z * q.imag.z)
        let yaw = atan2(sinYawCosPitch, cosYawCosPitch)
        return SIMD3<Float>(roll, pitch, yaw)
    }
}
