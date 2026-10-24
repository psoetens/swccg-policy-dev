package com.gempukku.swccgo.bench;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * JNI binding of swccg-policy's native/libswccgpolicy.so (libtorch + CUDA graphs): TorchScript policy slots and batched
 * forwards on padded bucket buffers (layout: {@link PolicyTensors#pack}). The library path comes from -Dpolicy.native.lib.
 * Not thread-safe per call site: the library serialises GPU work internally; PolicyEngine calls forward from one thread.
 */
public final class NativePolicy {
    private static boolean loaded;

    private NativePolicy() {
    }

    public static synchronized void ensureLoaded() {
        if (loaded) return;
        String lib = System.getProperty("policy.native.lib");
        if (lib == null) throw new IllegalStateException("-Dpolicy.native.lib=PATH/libswccgpolicy.so not set");
        System.load(lib);
        loaded = true;
    }

    public static native void setTf32(boolean on);

    public static native String lastError();

    /** Loads a TorchScript policy onto the GPU; 0 on failure (see {@link #lastError}). */
    public static native long load(String tsPath, boolean graphs);

    /** Copies the parameters and buffers of another .ts of the same architecture into the slot in place; 0 = ok. */
    public static native int swap(long slot, String tsPath);

    /** Forward of one padded batch (direct buffers): in = bucket layout for (b, t, o), out >= b * o floats; 0 = ok. */
    public static native int forward(long slot, ByteBuffer in, int b, int t, int o, FloatBuffer out);

    /** Queues the forward of one padded batch on the slot's stream and returns; {@link #collect} fetches the logits; 0 = ok. */
    public static native int launch(long slot, ByteBuffer in, int b, int t, int o);

    /** Waits (sleeping, not spinning) for the slot's pending launch and copies its logits (b * o floats) to out; 0 = ok. */
    public static native int collect(long slot, FloatBuffer out);

    public static native void free(long slot);
}
