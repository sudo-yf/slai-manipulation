import ARKit
import RealityKit
import SwiftUI

struct ARSceneView: UIViewRepresentable {
    let controller: ARCaptureController

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.environment.background = .cameraFeed()
        controller.attach(to: view.session)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}
