import Foundation
import Network
import simd

struct PosePacket: Encodable {
    let formatVersion: Int
    let sequence: UInt64
    let timestampSeconds: TimeInterval
    let sentAtUnixSeconds: TimeInterval
    let tracking: String
    let teleoperationEnabled: Bool
    let teleoperationEpoch: UInt64
    let worldFromCamera: [Float]

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case sequence
        case timestampSeconds = "timestamp_s"
        case sentAtUnixSeconds = "sent_at_unix_s"
        case tracking
        case teleoperationEnabled = "teleop_enabled"
        case teleoperationEpoch = "teleop_epoch"
        case worldFromCamera = "world_from_camera"
    }

    init(
        sequence: UInt64,
        timestampSeconds: TimeInterval,
        tracking: String,
        teleoperationEnabled: Bool,
        teleoperationEpoch: UInt64,
        transform: simd_float4x4
    ) {
        self.formatVersion = 2
        self.sequence = sequence
        self.timestampSeconds = timestampSeconds
        self.sentAtUnixSeconds = Date().timeIntervalSince1970
        self.tracking = tracking
        self.teleoperationEnabled = teleoperationEnabled
        self.teleoperationEpoch = teleoperationEpoch
        self.worldFromCamera = (0..<4).flatMap { row in
            (0..<4).map { column in transform[column][row] }
        }
    }
}

final class PoseTCPServer {
    private let queue = DispatchQueue(label: "com.sixdposetool.pose-stream")
    private let encoder = JSONEncoder()
    private let port: NWEndpoint.Port
    private var listener: NWListener?
    private var connection: NWConnection?
    private var connectionReady = false
    private var pendingData: Data?
    private var inFlightData: Data?
    private var isSending = false
    private var sentFrames = 0
    private var supersededFrames = 0

    var statusChanged: ((String) -> Void)?
    var countersChanged: ((Int, Int) -> Void)?

    init(port: UInt16 = 5005) {
        self.port = NWEndpoint.Port(rawValue: port)!
    }

    func start() {
        queue.async { [weak self] in
            guard let self, self.listener == nil else { return }
            do {
                let listener = try NWListener(using: .tcp, on: self.port)
                listener.newConnectionHandler = { [weak self] connection in
                    self?.accept(connection)
                }
                listener.stateUpdateHandler = { [weak self] state in
                    switch state {
                    case .ready:
                        self?.publishStatus("Awaiting USB receiver on TCP 5005")
                    case .failed(let error):
                        self?.publishStatus("Stream error: \(error.localizedDescription)")
                    case .cancelled:
                        self?.publishStatus("Stream stopped")
                    default:
                        break
                    }
                }
                self.listener = listener
                listener.start(queue: self.queue)
            } catch {
                self.publishStatus("Cannot open TCP 5005: \(error.localizedDescription)")
            }
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.connection?.cancel()
            self.listener?.cancel()
            self.connection = nil
            self.listener = nil
            self.connectionReady = false
            self.pendingData = nil
            self.inFlightData = nil
            self.isSending = false
            self.publishStatus("Stream stopped")
        }
    }

    func send(_ packet: PosePacket) {
        queue.async { [weak self] in
            guard let self, var data = try? self.encoder.encode(packet) else { return }
            data.append(0x0A)
            if self.pendingData != nil {
                self.supersededFrames += 1
                self.publishCounters()
            }
            self.pendingData = data
            self.drainLatestFrame()
        }
    }

    private func accept(_ newConnection: NWConnection) {
        connection?.cancel()
        connection = newConnection
        connectionReady = false
        newConnection.stateUpdateHandler = { [weak self, weak newConnection] state in
            guard let self else { return }
            switch state {
            case .ready:
                guard self.connection === newConnection else { return }
                self.connectionReady = true
                self.publishStatus("USB receiver connected")
                self.drainLatestFrame()
            case .failed(let error):
                self.disconnect(newConnection, message: "Stream connection error: \(error.localizedDescription)")
            case .cancelled:
                self.disconnect(newConnection, message: "Awaiting USB receiver on TCP 5005")
            default:
                break
            }
        }
        newConnection.start(queue: queue)
    }

    private func drainLatestFrame() {
        guard !isSending, connectionReady, let connection, let data = pendingData else { return }
        pendingData = nil
        inFlightData = data
        isSending = true
        connection.send(content: data, completion: .contentProcessed { [weak self, weak connection] error in
            guard let self else { return }
            self.queue.async {
                self.isSending = false
                let sentData = self.inFlightData
                self.inFlightData = nil
                if let error {
                    if self.pendingData == nil {
                        self.pendingData = sentData
                    } else {
                        self.supersededFrames += 1
                    }
                    self.disconnect(connection, message: "Stream send error: \(error.localizedDescription)")
                    return
                }
                self.sentFrames += 1
                self.publishCounters()
                self.drainLatestFrame()
            }
        })
    }

    private func disconnect(_ candidate: NWConnection?, message: String) {
        guard candidate == nil || connection === candidate else { return }
        connection?.cancel()
        connection = nil
        connectionReady = false
        publishStatus(message)
    }

    private func publishStatus(_ value: String) {
        DispatchQueue.main.async { [weak self] in
            self?.statusChanged?(value)
        }
    }

    private func publishCounters() {
        let sent = sentFrames
        let superseded = supersededFrames
        DispatchQueue.main.async { [weak self] in
            self?.countersChanged?(sent, superseded)
        }
    }
}
