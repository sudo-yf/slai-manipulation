import Foundation

final class PoseWebSocketPublisher {
    private struct Configuration: Equatable {
        let endpoint: String
        let sessionID: String
        let token: String
    }

    private let queue = DispatchQueue(label: "com.sixdposetool.cloud-stream")
    private let encoder = JSONEncoder()
    private let urlSession = URLSession(configuration: .default)
    private var configuration: Configuration?
    private var task: URLSessionWebSocketTask?
    private var authenticated = false
    private var isSending = false
    private var pendingMessage: URLSessionWebSocketTask.Message?
    private var sentFrames = 0
    private var supersededFrames = 0
    private var reconnectScheduled = false

    var statusChanged: ((String) -> Void)?
    var countersChanged: ((Int, Int) -> Void)?

    func start(endpoint: String, sessionID: String, token: String) {
        queue.async { [weak self] in
            guard let self else { return }
            let configuration = Configuration(endpoint: endpoint, sessionID: sessionID, token: token)
            guard self.configuration != configuration || self.task == nil else { return }
            self.stopConnection()
            self.configuration = configuration
            self.connect()
        }
    }

    func stop() {
        queue.async { [weak self] in
            self?.configuration = nil
            self?.stopConnection()
            self?.publishStatus("Cloud stream stopped")
        }
    }

    func send(_ packet: PosePacket) {
        queue.async { [weak self] in
            guard let self, let data = try? self.encoder.encode(packet) else { return }
            if self.pendingMessage != nil {
                self.supersededFrames += 1
                self.publishCounters()
            }
            self.pendingMessage = .data(data)
            self.drainLatestFrame()
        }
    }

    private func connect() {
        guard let configuration, let url = webSocketURL(for: configuration.endpoint, sessionID: configuration.sessionID) else {
            publishStatus("Cloud URL is invalid")
            return
        }
        let task = urlSession.webSocketTask(with: url)
        self.task = task
        self.authenticated = false
        task.resume()
        publishStatus("Connecting to cloud")
        let authentication = ["type": "auth", "token": configuration.token]
        guard let data = try? JSONSerialization.data(withJSONObject: authentication),
              let text = String(data: data, encoding: .utf8) else {
            publishStatus("Cloud authentication setup failed")
            return
        }
        task.send(.string(text)) { [weak self, weak task] error in
            self?.queue.async {
                guard let self, let task, self.task === task else { return }
                if let error {
                    self.connectionFailed(error)
                } else {
                    self.receive(on: task)
                }
            }
        }
    }

    private func receive(on task: URLSessionWebSocketTask) {
        task.receive { [weak self, weak task] result in
            self?.queue.async {
                guard let self, self.task === task else { return }
                switch result {
                case .success(.string(let text)):
                    if let payload = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
                       payload["type"] as? String == "authenticated" {
                        self.authenticated = true
                        self.publishStatus("Cloud stream connected")
                        self.drainLatestFrame()
                    }
                    if let task { self.receive(on: task) }
                case .success:
                    if let task { self.receive(on: task) }
                case .failure(let error):
                    self.connectionFailed(error)
                }
            }
        }
    }

    private func drainLatestFrame() {
        guard authenticated, !isSending, let task, let message = pendingMessage else { return }
        pendingMessage = nil
        isSending = true
        task.send(message) { [weak self, weak task] error in
            self?.queue.async {
                guard let self, self.task === task else { return }
                self.isSending = false
                if let error {
                    self.connectionFailed(error)
                    return
                }
                self.sentFrames += 1
                self.publishCounters()
                self.drainLatestFrame()
            }
        }
    }

    private func connectionFailed(_ error: Error) {
        guard configuration != nil else { return }
        stopConnection()
        publishStatus("Cloud reconnecting")
        guard !reconnectScheduled else { return }
        reconnectScheduled = true
        queue.asyncAfter(deadline: .now() + 2) { [weak self] in
            guard let self else { return }
            self.reconnectScheduled = false
            self.connect()
        }
    }

    private func stopConnection() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        authenticated = false
        isSending = false
    }

    private func webSocketURL(for endpoint: String, sessionID: String) -> URL? {
        var components = URLComponents(string: endpoint.trimmingCharacters(in: .whitespacesAndNewlines))
        if components?.scheme == "https" { components?.scheme = "wss" }
        if components?.scheme == "http" { components?.scheme = "ws" }
        let basePath = components?.path.trimmingCharacters(in: CharacterSet(charactersIn: "/")) ?? ""
        let encodedSessionID = sessionID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? sessionID
        components?.path = "/" + ([basePath, "ws", "ingest", encodedSessionID].filter { !$0.isEmpty }.joined(separator: "/"))
        return components?.url
    }

    private func publishStatus(_ value: String) {
        DispatchQueue.main.async { [weak self] in self?.statusChanged?(value) }
    }

    private func publishCounters() {
        let sent = sentFrames
        let superseded = supersededFrames
        DispatchQueue.main.async { [weak self] in self?.countersChanged?(sent, superseded) }
    }
}
