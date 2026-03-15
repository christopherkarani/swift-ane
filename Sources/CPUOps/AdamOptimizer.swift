import Accelerate

public enum AdamOptimizer {
    public struct Workspace: ~Copyable {
        public let capacity: Int
        fileprivate let tmp1: UnsafeMutablePointer<Float>
        fileprivate let tmp2: UnsafeMutablePointer<Float>

        public init(capacity: Int) {
            precondition(capacity >= 0)
            self.capacity = capacity
            self.tmp1 = .allocate(capacity: capacity)
            self.tmp2 = .allocate(capacity: capacity)
        }

        deinit {
            tmp1.deallocate()
            tmp2.deallocate()
        }
    }

    /// In-place Adam update with bias correction. Mutates w, m, v.
    public static func update(
        weights: UnsafeMutablePointer<Float>,
        gradients: UnsafePointer<Float>,
        m: UnsafeMutablePointer<Float>,
        v: UnsafeMutablePointer<Float>,
        count: Int,
        timestep: Int, // starts at 1, NOT 0
        lr: Float,
        beta1: Float,
        beta2: Float,
        eps: Float
    ) {
        precondition(count >= 0)
        precondition(timestep >= 1)

        let t = Float(timestep)
        let bc1 = 1.0 - powf(beta1, t)
        let bc2 = 1.0 - powf(beta2, t)

        for i in 0..<count {
            let g = gradients[i]
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g
            let mh = m[i] / bc1
            let vh = v[i] / bc2
            weights[i] -= lr * mh / (sqrtf(vh) + eps)
        }
    }

    public static func update(
        weights: UnsafeMutablePointer<Float>,
        gradients: UnsafePointer<Float>,
        m: UnsafeMutablePointer<Float>,
        v: UnsafeMutablePointer<Float>,
        count: Int,
        timestep: Int,
        lr: Float,
        beta1: Float,
        beta2: Float,
        eps: Float,
        workspace: borrowing Workspace
    ) {
        precondition(count >= 0)
        precondition(timestep >= 1)
        precondition(workspace.capacity >= count)

        if count == 0 {
            return
        }

        let t = Float(timestep)
        let bc1 = 1.0 - powf(beta1, t)
        let bc2 = 1.0 - powf(beta2, t)

        let n = vDSP_Length(count)
        let tmp1 = workspace.tmp1
        let tmp2 = workspace.tmp2

        var beta1Scale = beta1
        var oneMinusBeta1: Float = 1.0 - beta1
        vDSP_vsmul(m, 1, &beta1Scale, m, 1, n)
        vDSP_vsma(gradients, 1, &oneMinusBeta1, m, 1, m, 1, n)

        var beta2Scale = beta2
        var oneMinusBeta2: Float = 1.0 - beta2
        vDSP_vsq(gradients, 1, tmp1, 1, n)
        vDSP_vsmul(v, 1, &beta2Scale, v, 1, n)
        vDSP_vsma(tmp1, 1, &oneMinusBeta2, v, 1, v, 1, n)

        var biasCorrection2 = bc2
        vDSP_vsdiv(v, 1, &biasCorrection2, tmp1, 1, n)

        var count32 = Int32(count)
        vvsqrtf(tmp1, tmp1, &count32)

        var epsilon = eps
        vDSP_vsadd(tmp1, 1, &epsilon, tmp1, 1, n)

        var biasCorrection1 = bc1
        vDSP_vsdiv(m, 1, &biasCorrection1, tmp2, 1, n)
        vDSP_vdiv(tmp1, 1, tmp2, 1, tmp1, 1, n)

        var negLr = -lr
        vDSP_vsma(tmp1, 1, &negLr, weights, 1, weights, 1, n)
    }
}
