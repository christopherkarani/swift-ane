import Foundation
import ANETypes

enum RWKVStyleRecurrentOutputTouch {
    static func append(
        to builder: inout MILBuilder,
        dim: Int,
        lane: Int,
        prefix: String,
        inputName: String,
        stateName: String,
        projName: String,
        rawOutputName: String,
        rawStateName: String,
        outputName: String,
        outputStateName: String
    ) {
        let inputChannel = "\(prefix)in_ch"
        let stateChannel = "\(prefix)state_ch"
        let projChannel = "\(prefix)proj_ch"
        let inputScalar = "\(prefix)in_s"
        let stateScalar = "\(prefix)state_s"
        let projScalar = "\(prefix)proj_s"
        let touch0 = "\(prefix)touch0"
        let touch = "\(prefix)touch"
        let zero = "\(prefix)z"

        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(inputChannel) = reduce_sum(x=\(inputName),axes=raxCh,keep_dims=kd)[name=string(\"\(inputChannel)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(stateChannel) = reduce_sum(x=\(stateName),axes=raxCh,keep_dims=kd)[name=string(\"\(stateChannel)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,\(lane)]> \(projChannel) = reduce_sum(x=\(projName),axes=raxCh,keep_dims=kd)[name=string(\"\(projChannel)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,1]> \(inputScalar) = reduce_sum(x=\(inputChannel),axes=raxSp,keep_dims=kd)[name=string(\"\(inputScalar)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,1]> \(stateScalar) = reduce_sum(x=\(stateChannel),axes=raxSp,keep_dims=kd)[name=string(\"\(stateScalar)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,1]> \(projScalar) = reduce_sum(x=\(projChannel),axes=raxSp,keep_dims=kd)[name=string(\"\(projScalar)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,1]> \(touch0) = add(x=\(inputScalar),y=\(stateScalar))[name=string(\"\(touch0)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,1]> \(touch) = add(x=\(touch0),y=\(projScalar))[name=string(\"\(touch)\")];")
        builder.appendLine("        tensor<fp16, [1,1,1,1]> \(zero) = mul(x=\(touch),y=zc)[name=string(\"\(zero)\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputStateName) = add(x=\(rawStateName),y=\(zero))[name=string(\"\(outputStateName)\")];")
        builder.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> \(outputName) = add(x=\(rawOutputName),y=\(zero))[name=string(\"\(outputName)\")];")
    }
}
