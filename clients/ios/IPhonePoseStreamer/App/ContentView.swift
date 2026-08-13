import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var controller = ARCaptureController()
    @State private var didCopyPose = false
    @State private var showingCloudSettings = false

    var body: some View {
        ZStack(alignment: .bottom) {
            ARSceneView(controller: controller)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("6D Pose Tool", systemImage: "viewfinder")
                        .font(.headline)
                    Spacer()
                    Text(controller.isRunning ? "LIVE" : "IDLE")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(controller.isRunning ? .green : .secondary)
                    Button {
                        showingCloudSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Cloud stream settings")
                }

                Divider()

                statusRow("Tracking", controller.trackingDescription)
                statusRow("Depth", controller.depthAvailable ? "Enabled" : "Disabled for low latency")
                statusRow("Frame rate", "\(controller.frameRate) fps")
                statusRow("TCP stream", controller.transportDescription)
                statusRow("Stream frames", "sent \(controller.sentFrameCount) / superseded \(controller.supersededFrameCount)")
                statusRow("Cloud stream", controller.cloudDescription)
                statusRow("Cloud frames", "sent \(controller.cloudSentFrameCount) / superseded \(controller.cloudSupersededFrameCount)")
                statusRow("Teleoperation", controller.teleoperationEnabled ? "Enabled" : "Hold")
                poseRow("Position (m)", controller.position, precision: 3)
                poseRow("Rotation (deg)", controller.orientationDegrees, precision: 1)

                Label(
                    controller.teleoperationEnabled ? "MOVING - RELEASE TO HOLD" : "HOLD TO MOVE",
                    systemImage: controller.teleoperationEnabled ? "hand.raised.fill" : "hand.point.up.left.fill"
                )
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(controller.teleoperationEnabled ? Color.white : Color.primary)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(controller.teleoperationEnabled ? Color.red : Color.secondary.opacity(0.18))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
                .onLongPressGesture(
                    minimumDuration: .infinity,
                    maximumDistance: 80,
                    pressing: { controller.setTeleoperationEnabled($0) },
                    perform: {}
                )
                .opacity(controller.isRunning ? 1 : 0.45)
                .accessibilityLabel("Hold to enable teleoperation")

                HStack(spacing: 12) {
                    Button(controller.isRunning ? "Stop" : "Start") {
                        controller.isRunning ? controller.stop() : controller.start()
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        controller.recenter()
                    } label: {
                        Label("Recenter", systemImage: "arrow.counterclockwise")
                    }
                    .buttonStyle(.bordered)
                    .disabled(!controller.isRunning)

                    Button {
                        UIPasteboard.general.string = controller.posePayload()
                        didCopyPose = true
                    } label: {
                        Label(didCopyPose ? "Copied" : "Copy pose", systemImage: didCopyPose ? "checkmark" : "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                    .disabled(!controller.isRunning)

                    Button {
                        controller.openViewer()
                    } label: {
                        Image(systemName: "globe")
                    }
                    .buttonStyle(.bordered)
                    .disabled(controller.cloudViewerURL.isEmpty)
                    .accessibilityLabel("Open pose viewer")
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            .padding()
        }
        .onAppear { controller.start() }
        .sheet(isPresented: $showingCloudSettings) {
            CloudSettingsView(controller: controller)
        }
    }

    private func statusRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value).multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }

    private func poseRow(_ label: String, _ value: SIMD3<Float>, precision: Int) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(String(format: "[%.*f, %.*f, %.*f]", precision, value.x, precision, value.y, precision, value.z))
                .monospacedDigit()
        }
        .font(.subheadline)
    }
}

private struct CloudSettingsView: View {
    @ObservedObject var controller: ARCaptureController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Pose Hub") {
                    TextField("WebSocket URL", text: $controller.cloudEndpoint)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    TextField("Session ID", text: $controller.cloudSessionID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Ingest token", text: $controller.cloudToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Viewer URL", text: $controller.cloudViewerURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
            }
            .navigationTitle("Cloud Stream")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        controller.saveCloudSettings()
                        dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
