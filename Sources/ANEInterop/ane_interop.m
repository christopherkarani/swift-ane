#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dispatch/dispatch.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ane_interop.h"

struct ANEHandle {
    void *model;               // CFBridgingRetain'd _ANEInMemoryModel
    void *client;              // CFBridgingRetain'd _ANEClient (optional)
    void *clientModel;         // CFBridgingRetain'd _ANEModel (optional)
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    void *request;             // CFBridgingRetain'd _ANERequest
    void *perfStats;           // CFBridgingRetain'd _ANEPerformanceStats (optional)
    bool perfStatsRequested;
    unsigned int perfStatsMask;
    void *evalOptions;         // CFBridgingRetain'd NSDictionary (optional)
    bool realtimeLoaded;
    void *tmpDir;              // CFBridgingRetain'd NSString
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    bool liveHandleCounted;
    uint64_t lastHwExecutionTimeNS;
};

static Class g_ANEDesc = nil, g_ANEInMem = nil, g_ANEReq = nil, g_ANEIO = nil;
static bool g_ane_loaded = false;
static dispatch_once_t g_ane_once;
static int g_compile_count = 0;
static int g_last_compile_error = ANE_INTEROP_COMPILE_ERROR_NONE;
static int g_force_eval_failure = 0;
static int g_live_handle_count = 0;

static bool ane_interop_trace_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = (getenv("ANE_INTEROP_TRACE") != NULL) ? 1 : 0;
    }
    return cached == 1;
}

static bool ane_interop_perf_stats_enabled(void) {
    // Do not cache: tests and benchmarks may toggle this env var at runtime.
    const char *v = getenv("ANE_PERF_STATS");
    return (v && v[0] == '1');
}

static unsigned int ane_interop_perf_stats_anef_mask(void) {
    // _ANEPerformanceStats.driverMaskForANEFMask: appears to only accept a subset of bits.
    // On M3 Max, any bits outside {0x1,0x2,0x4,0x8} cause it to return 0 (disabling stats).
    // Default to 0xF (enable all supported perf-stats features) and allow override for probing.
    const char *v = getenv("ANE_PERF_STATS_MASK");
    if (!v || v[0] == '\0') return 0xFu;
    char *end = NULL;
    unsigned long parsed = strtoul(v, &end, 0);
    if (end == v) return 0xFu;
    return (unsigned int)parsed;
}

static bool ane_interop_env_flag(const char *key) {
    const char *v = getenv(key);
    return (v && v[0] == '1');
}

static long ane_interop_env_long(const char *key, long defaultValue) {
    const char *v = getenv(key);
    if (!v || v[0] == '\0') return defaultValue;
    char *end = NULL;
    long parsed = strtol(v, &end, 0);
    if (end == v) return defaultValue;
    return parsed;
}

typedef enum : int {
    ANE_COMPILE_CACHE_AUTO = 0,
    ANE_COMPILE_CACHE_PREFER_CACHED = 1,
    ANE_COMPILE_CACHE_FORCE_COLD = 2,
} ANECompileCachePolicy;

static ANECompileCachePolicy ane_interop_compile_cache_policy(void) {
    const char *v = getenv("ANE_COMPILE_CACHE_POLICY");
    if (!v || v[0] == '\0') return ANE_COMPILE_CACHE_AUTO;
    if (strcmp(v, "preferCached") == 0 || strcmp(v, "prefer_cached") == 0) return ANE_COMPILE_CACHE_PREFER_CACHED;
    if (strcmp(v, "forceCold") == 0 || strcmp(v, "force_cold") == 0) return ANE_COMPILE_CACHE_FORCE_COLD;
    return ANE_COMPILE_CACHE_AUTO;
}

typedef enum : int {
    ANE_EVAL_INMEM = 0,
    ANE_EVAL_CLIENT = 1,
    ANE_EVAL_CLIENT_DIRECT = 2,
    ANE_EVAL_REALTIME = 3,
} ANEEvalPath;

static ANEEvalPath ane_interop_eval_path(void) {
    const char *v = getenv("ANE_EVAL_PATH");
    if (!v || v[0] == '\0') return ANE_EVAL_INMEM;
    if (strcmp(v, "client") == 0) return ANE_EVAL_CLIENT;
    if (strcmp(v, "clientDirect") == 0 || strcmp(v, "client_direct") == 0) return ANE_EVAL_CLIENT_DIRECT;
    if (strcmp(v, "realtime") == 0 || strcmp(v, "realTime") == 0) return ANE_EVAL_REALTIME;
    return ANE_EVAL_INMEM;
}

static void ane_interop_set_compile_error(int value) {
    __sync_lock_test_and_set(&g_last_compile_error, value);
}

static bool ane_interop_size_mul_overflow(size_t a, size_t b, size_t *out) {
#if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
#endif
}

void ane_interop_init(void) {
    if (g_ane_loaded) return;
    dispatch_once(&g_ane_once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        g_ANEReq = NSClassFromString(@"_ANERequest");
        g_ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_ane_loaded = true;
    });
}

IOSurfaceRef ane_interop_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static void ane_interop_remove_tmpdir(NSString *td) {
    if (!td) return;
    if (getenv("ANE_KEEP_TMPDIR") != NULL) return;
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
}

static NSString *ane_interop_sanitized_relative_weight_path(NSString *path) {
    static NSString * const kModelPrefix = @"@model_path/";
    if (![path hasPrefix:kModelPrefix]) return nil;

    NSString *rel = [path substringFromIndex:kModelPrefix.length];
    if (rel.length == 0 || [rel hasPrefix:@"/"]) return nil;

    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    for (NSString *comp in [rel pathComponents]) {
        if (comp.length == 0 || [comp isEqualToString:@"/"]) continue;
        if ([comp isEqualToString:@"."] || [comp isEqualToString:@".."]) return nil;
        [parts addObject:comp];
    }
    if (parts.count == 0) return nil;
    return [NSString pathWithComponents:parts];
}

ANEHandle *ane_interop_compile(const uint8_t *milText, size_t milLen,
                               const char **weightPaths, const uint8_t **weightDatas,
                               const size_t *weightLens, int weightCount,
                               int nInputs, const size_t *inputSizes,
                               int nOutputs, const size_t *outputSizes) {
    @autoreleasepool {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
        if (!milText || milLen == 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (weightCount < 0 || nInputs < 0 || nOutputs < 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (weightCount > 0 && (!weightPaths || !weightDatas || !weightLens)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (nInputs > 0 && !inputSizes) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (nOutputs > 0 && !outputSizes) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }

        ane_interop_init();
        if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        NSData *milData = [NSData dataWithBytesNoCopy:(void *)milText length:milLen freeWhenDone:NO];
        if (!milData) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        NSMutableDictionary *weights = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)weightCount];
        for (int i = 0; i < weightCount; i++) {
            if (!weightPaths[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }
            if (weightLens[i] > 0 && !weightDatas[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }

            NSString *path = [NSString stringWithUTF8String:weightPaths[i]];
            if (!path) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }
            if (weights[path] != nil) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH);
                return NULL;
            }

            NSData *wd = [NSData dataWithBytesNoCopy:(void *)weightDatas[i]
                                              length:weightLens[i]
                                        freeWhenDone:NO];
            if (!wd) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }
            weights[path] = @{@"offset": @0, @"data": wd};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, (id)(weightCount ? weights : @{}), nil);
        if (!desc) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        const bool wantPerfStats = ane_interop_perf_stats_enabled();
        id perfStats = nil;
        if (wantPerfStats) {
            Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
            SEL makeSel = @selector(statsWithRequestPerformanceBuffer:statsBufferSize:);
            if (perfClass && [perfClass respondsToSelector:makeSel]) {
                void *buf = NULL;
                unsigned int bufSize = 0;
                perfStats = ((id(*)(Class,SEL,void **, unsigned int *))objc_msgSend)(
                    perfClass, makeSel, &buf, &bufSize);
                if (ane_interop_trace_enabled()) {
                    fprintf(stderr, "ANE perfStats factory: %s (buf=%p bufSize=%u)\n", perfStats ? "OK" : "nil", buf, bufSize);
                }
            } else if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE perfStats: class/method unavailable\n");
            }
        }

        ANECompileCachePolicy cachePolicy = ane_interop_compile_cache_policy();
        BOOL compiledExists = NO;
        if ([mdl respondsToSelector:@selector(compiledModelExists)]) {
            compiledExists = ((BOOL(*)(id,SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
        }
        if (cachePolicy == ANE_COMPILE_CACHE_FORCE_COLD && [mdl respondsToSelector:@selector(purgeCompiledModel)]) {
            ((void(*)(id,SEL))objc_msgSend)(mdl, @selector(purgeCompiledModel));
            compiledExists = NO;
        }

        unsigned int perfMask = 0;
        NSMutableDictionary *baseOptions = nil;
        if (wantPerfStats || ane_interop_env_flag("ANE_DISABLE_POWER_SAVING") ||
            ane_interop_env_flag("ANE_KEEP_MODEL_WIRED") || ane_interop_env_flag("ANE_ENABLE_LATE_LATCH") ||
            ane_interop_env_flag("ANE_SKIP_PREPARE") || ane_interop_env_flag("ANE_ENABLE_FW_TO_FW_SIGNAL") ||
            ane_interop_env_flag("ANE_DISABLE_IO_FENCES") || getenv("ANE_MEMORY_POOL_ID") != NULL) {
            baseOptions = [NSMutableDictionary dictionary];
        }

        if (baseOptions && ane_interop_env_flag("ANE_DISABLE_POWER_SAVING")) {
            baseOptions[@"kANEFEnablePowerSavingKey"] = @NO;
        }
        if (baseOptions && ane_interop_env_flag("ANE_KEEP_MODEL_WIRED")) {
            baseOptions[@"kANEFKeepModelMemoryWiredKey"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_ENABLE_LATE_LATCH")) {
            baseOptions[@"kANEFEnableLateLatchKey"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_SKIP_PREPARE")) {
            baseOptions[@"kANEFSkipPreparePhaseKey"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_ENABLE_FW_TO_FW_SIGNAL")) {
            baseOptions[@"kANEFEnableFWToFWSignal"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_DISABLE_IO_FENCES")) {
            baseOptions[@"kANEFDisableIOFencesUseSharedEventsKey"] = @YES;
        }
        if (baseOptions && getenv("ANE_MEMORY_POOL_ID") != NULL) {
            long mp = ane_interop_env_long("ANE_MEMORY_POOL_ID", -1);
            if (mp >= 0) {
                baseOptions[@"kANEFMemoryPoolIDKey"] = @(mp);
            }
        }

        if (wantPerfStats && [mdl respondsToSelector:@selector(setPerfStatsMask:)]) {
            // Enable perf stats collection (required for hwExecutionTime to populate).
            //
            // NOTE: driverMaskForANEFMask: returns 0 if any unsupported bits are set, so we
            // must be careful to only request supported ANEF bits.
            perfMask = ane_interop_perf_stats_anef_mask();
            Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
            SEL driverSel = @selector(driverMaskForANEFMask:);
            if (perfClass && [perfClass respondsToSelector:driverSel]) {
                perfMask = ((unsigned int(*)(Class,SEL,unsigned int))objc_msgSend)(perfClass, driverSel, perfMask);
            }
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE perfStatsMask: 0x%08X\n", perfMask);
            }
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), perfMask);

            // Attempt to also enable stats via driver option keys (best-effort; ignored if unknown).
            if (!baseOptions) {
                baseOptions = [NSMutableDictionary dictionary];
            }
            baseOptions[@"kANEFPerformanceStatsMask"] = @(perfMask);
            baseOptions[@"kANEFModelLoadPerformanceStats"] = @YES;
        }

        NSDictionary *finalOptions = baseOptions ? [baseOptions copy] : @{};
        if (ane_interop_env_flag("ANE_USE_COMPILER_OPTIONS") &&
            [mdl respondsToSelector:@selector(compilerOptionsWithOptions:isCompiledModelCached:)]) {
            id computed = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
                mdl, @selector(compilerOptionsWithOptions:isCompiledModelCached:), finalOptions, compiledExists);
            if ([computed isKindOfClass:[NSDictionary class]]) {
                finalOptions = (NSDictionary *)computed;
            }
        }
        if (ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE compile cachePolicy=%d compiledExists=%d options=%lu\n",
                    cachePolicy, (int)compiledExists, (unsigned long)[finalOptions count]);
        }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        if (![hx isKindOfClass:[NSString class]]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if (ane_interop_trace_enabled()) {
            fprintf(
                stderr,
                "ANE compile: id=%s tmpdir=%s milLen=%zu weights=%d inputs=%d outputs=%d\n",
                [((NSString *)hx) UTF8String],
                [td UTF8String],
                milLen,
                weightCount,
                nInputs,
                nOutputs
            );
            for (NSString *path in weights) {
                fprintf(stderr, "  weight: %s (%zu bytes)\n", [path UTF8String], [weights[path][@"data"] length]);
            }
        }
        NSFileManager *fm = [NSFileManager defaultManager];
        if (![fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
           withIntermediateDirectories:YES attributes:nil error:nil]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }
        if (![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_remove_tmpdir(td);
            return NULL;
        }
        NSString *tdStd = [td stringByStandardizingPath];
        NSString *tdPrefix = [tdStd hasSuffix:@"/"] ? tdStd : [tdStd stringByAppendingString:@"/"];
        for (NSString *path in weights) {
            NSString *rel = ane_interop_sanitized_relative_weight_path(path);
            if (!rel) {
                fprintf(stderr, "ANE compile failed: invalid weight path '%s'\n", [path UTF8String]);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            NSString *full = [[td stringByAppendingPathComponent:rel] stringByStandardizingPath];
            if (![full hasPrefix:tdPrefix]) {
                fprintf(stderr, "ANE compile failed: escaped tmp dir for weight path '%s'\n", [path UTF8String]);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            if (![fm createDirectoryAtPath:[full stringByDeletingLastPathComponent]
               withIntermediateDirectories:YES attributes:nil error:nil]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            if (![weights[path][@"data"] writeToFile:full atomically:YES]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
        }

        NSError *e = nil;
        if (!(cachePolicy == ANE_COMPILE_CACHE_PREFER_CACHED && compiledExists)) {
            if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(compileWithQoS:options:error:), 21, finalOptions, &e)) {
                // Retry without options (some host builds reject unknown keys).
                if ([finalOptions count] > 0) {
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE compile retrying without options...\n");
                    }
                    e = nil;
                    if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                        finalOptions = @{};
                    } else {
                        fprintf(stderr, "ANE compile failed: %s\n", e ? [[e description] UTF8String] : "no error");
                        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                        ane_interop_remove_tmpdir(td);
                        return NULL;
                    }
                } else {
                    fprintf(stderr, "ANE compile failed: %s\n", e ? [[e description] UTF8String] : "no error");
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                    ane_interop_remove_tmpdir(td);
                    return NULL;
                }
            }
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, finalOptions, &e)) {
            // Retry without options (keep behavior symmetric with compile).
            if ([finalOptions count] > 0) {
                if (ane_interop_trace_enabled()) {
                    fprintf(stderr, "ANE load retrying without options...\n");
                }
                e = nil;
                if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
                    finalOptions = @{};
                } else {
                    fprintf(stderr, "ANE load failed: %s\n", e ? [[e description] UTF8String] : "no error");
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                    ane_interop_remove_tmpdir(td);
                    return NULL;
                }
            } else {
                fprintf(stderr, "ANE load failed: %s\n", e ? [[e description] UTF8String] : "no error");
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
        }

        long qd = ane_interop_env_long("ANE_QUEUE_DEPTH", -1);
        if (qd >= 0 && [mdl respondsToSelector:@selector(setQueueDepth:)]) {
            if (qd > 127) qd = 127;
            ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), (char)qd);
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE queueDepth set to %ld\n", qd);
            }
        }

        id client = nil;
        id clientModel = nil;
        if ([mdl respondsToSelector:@selector(sharedConnection)]) {
            client = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(sharedConnection));
        }
        if ([mdl respondsToSelector:@selector(model)]) {
            clientModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        }

        const bool wantsRealtime = (ane_interop_eval_path() == ANE_EVAL_REALTIME);
        BOOL realtimeLoaded = NO;
        if (wantsRealtime && client && clientModel) {
            NSError *rtErr = nil;
            if ([client respondsToSelector:@selector(beginRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
            }
            if ([client respondsToSelector:@selector(loadRealTimeModel:options:qos:error:)]) {
                realtimeLoaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(loadRealTimeModel:options:qos:error:), clientModel, finalOptions, 21, &rtErr);
            }
            if (!realtimeLoaded && ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE realtime load failed: %s\n", rtErr ? [[rtErr description] UTF8String] : "no error");
            }
            if (!realtimeLoaded && [client respondsToSelector:@selector(endRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
            }
        }

        ANEHandle *h = (ANEHandle *)calloc(1, sizeof(ANEHandle));
        if (!h) {
            fprintf(stderr, "ANE compile failed: OOM allocating ANEHandle\n");
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            ane_interop_remove_tmpdir(td);
            return NULL;
        }
        h->model = (void *)CFBridgingRetain(mdl);
        h->client = client ? (void *)CFBridgingRetain(client) : NULL;
        h->clientModel = clientModel ? (void *)CFBridgingRetain(clientModel) : NULL;
        h->perfStats = perfStats ? (void *)CFBridgingRetain(perfStats) : NULL;
        h->perfStatsRequested = wantPerfStats;
        h->perfStatsMask = perfMask;
        h->evalOptions = (void *)CFBridgingRetain(finalOptions);
        h->realtimeLoaded = realtimeLoaded ? true : false;
        h->tmpDir = (void *)CFBridgingRetain(td);
        h->nInputs = nInputs;
        h->nOutputs = nOutputs;
        h->lastHwExecutionTimeNS = 0;

        if (nInputs > 0) {
            size_t inputMetaBytes = 0;
            if (ane_interop_size_mul_overflow((size_t)nInputs, sizeof(size_t), &inputMetaBytes)) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            h->inputBytes = (size_t *)malloc(inputMetaBytes);
            h->ioInputs = (IOSurfaceRef *)calloc((size_t)nInputs, sizeof(IOSurfaceRef));
            if (!h->inputBytes || !h->ioInputs) {
                fprintf(stderr, "ANE compile failed: OOM allocating input metadata\n");
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            memcpy(h->inputBytes, inputSizes, inputMetaBytes);
            for (int i = 0; i < nInputs; i++) {
                h->ioInputs[i] = ane_interop_create_surface(inputSizes[i]);
                if (!h->ioInputs[i]) {
                    fprintf(stderr, "ANE compile failed: IOSurfaceCreate returned NULL (input %d)\n", i);
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                    ane_interop_free(h);
                    return NULL;
                }
            }
        }
        if (nOutputs > 0) {
            size_t outputMetaBytes = 0;
            if (ane_interop_size_mul_overflow((size_t)nOutputs, sizeof(size_t), &outputMetaBytes)) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            h->outputBytes = (size_t *)malloc(outputMetaBytes);
            h->ioOutputs = (IOSurfaceRef *)calloc((size_t)nOutputs, sizeof(IOSurfaceRef));
            if (!h->outputBytes || !h->ioOutputs) {
                fprintf(stderr, "ANE compile failed: OOM allocating output metadata\n");
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            memcpy(h->outputBytes, outputSizes, outputMetaBytes);
            for (int i = 0; i < nOutputs; i++) {
                h->ioOutputs[i] = ane_interop_create_surface(outputSizes[i]);
                if (!h->ioOutputs[i]) {
                    fprintf(stderr, "ANE compile failed: IOSurfaceCreate returned NULL (output %d)\n", i);
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                    ane_interop_free(h);
                    return NULL;
                }
            }
        }

        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:(NSUInteger)nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)nInputs];
        for (int i = 0; i < nInputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), h->ioInputs[i]);
            if (!obj) {
                fprintf(stderr, "ANE compile failed: _ANEIOSurfaceObject returned nil (input %d)\n", i);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                ane_interop_free(h);
                return NULL;
            }
            [wIns addObject:obj];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:(NSUInteger)nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)nOutputs];
        for (int i = 0; i < nOutputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), h->ioOutputs[i]);
            if (!obj) {
                fprintf(stderr, "ANE compile failed: _ANEIOSurfaceObject returned nil (output %d)\n", i);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                ane_interop_free(h);
                return NULL;
            }
            [wOuts addObject:obj];
            [oIdx addObject:@(i)];
        }

        id req = nil;
        SEL reqSelPerfWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
        SEL reqSelWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:);
        if (perfStats) {
            if ([g_ANEReq respondsToSelector:reqSelPerfWB]) {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelPerfWB, wIns, iIdx, wOuts, oIdx, nil, perfStats, @0);
            }
            if (!req) {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:),
                    wIns, iIdx, wOuts, oIdx, perfStats, @0);
            }
        } else {
            // If perfStats factory is unavailable but perfStatsMask is set, the driver may attach perfStatsArray.
            if ([g_ANEReq respondsToSelector:reqSelWB]) {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelWB, wIns, iIdx, wOuts, oIdx, nil, @0);
            }
            if (!req) {
                req = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
                    wIns, iIdx, wOuts, oIdx, @0);
            }
        }
        if (!req) {
            fprintf(stderr, "ANE compile failed: _ANERequest returned nil\n");
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_free(h);
            return NULL;
        }
        h->request = (void *)CFBridgingRetain(req);

        h->liveHandleCounted = true;
        __sync_fetch_and_add(&g_live_handle_count, 1);
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
        __sync_fetch_and_add(&g_compile_count, 1);
        return h;
    }
}

bool ane_interop_eval(ANEHandle *handle) {
    if (!handle) return false;
    if (__sync_fetch_and_add(&g_force_eval_failure, 0) != 0) {
        return false;
    }
    id mdl = (__bridge id)handle->model;
    id req = (__bridge id)handle->request;
    NSError *e = nil;
    NSDictionary *options = handle->evalOptions ? (__bridge NSDictionary *)handle->evalOptions : @{};

    BOOL ok = NO;
    ANEEvalPath evalPath = ane_interop_eval_path();
    if (evalPath == ANE_EVAL_REALTIME && !handle->realtimeLoaded) {
        // Real-time path may be unavailable on public builds; fall back to standard in-memory eval.
        evalPath = ANE_EVAL_INMEM;
    }
    const bool shouldTryClient = (evalPath != ANE_EVAL_INMEM) || handle->perfStatsRequested;
    if (shouldTryClient && handle->client && handle->clientModel) {
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        if (evalPath == ANE_EVAL_REALTIME && handle->realtimeLoaded &&
            [client respondsToSelector:@selector(evaluateRealTimeWithModel:options:request:error:)]) {
            ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                client, @selector(evaluateRealTimeWithModel:options:request:error:), modelObj, options, req, &e);
        } else {
            SEL sel = @selector(evaluateWithModel:options:request:qos:error:);
            if (evalPath == ANE_EVAL_CLIENT_DIRECT || handle->perfStatsRequested) {
                SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
                if ([client respondsToSelector:directSel]) {
                    sel = directSel;
                }
            }
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, sel, modelObj, options, req, 21, &e);
        }
        if (!ok && ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE client eval failed (will fallback): %s\n", e ? [[e description] UTF8String] : "no error");
        }
    }
    if (!ok) {
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, options, req, &e);
    }
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n", e ? [[e description] UTF8String] : "no error");
        handle->lastHwExecutionTimeNS = 0;
    } else if (handle->perfStatsRequested) {
        id ps = nil;
        // Some drivers appear to replace/attach perf stats on the request after eval.
        if ([req respondsToSelector:@selector(perfStats)]) {
            ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
        }
        if (!ps && [req respondsToSelector:@selector(perfStatsArray)]) {
            id arr = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
            if ([arr isKindOfClass:[NSArray class]] && [arr count] > 0) {
                ps = [arr objectAtIndex:0];
            }
        }
        if (!ps) {
            ps = handle->perfStats ? (__bridge id)handle->perfStats : nil;
        }
        handle->lastHwExecutionTimeNS = ps ? ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime)) : 0;
        if (ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE hwExecutionTime: %llu ns\n", (unsigned long long)handle->lastHwExecutionTimeNS);
        }
    }
    return ok;
}

IOSurfaceRef ane_interop_get_input(ANEHandle *handle, int index) {
    if (!handle) return NULL;
    if (index < 0 || index >= handle->nInputs) return NULL;
    return handle->ioInputs[index];
}

IOSurfaceRef ane_interop_get_output(ANEHandle *handle, int index) {
    if (!handle) return NULL;
    if (index < 0 || index >= handle->nOutputs) return NULL;
    return handle->ioOutputs[index];
}

IOSurfaceRef ane_interop_copy_input(ANEHandle *handle, int index) {
    IOSurfaceRef s = ane_interop_get_input(handle, index);
    if (!s) return NULL;
    CFRetain(s);
    return s;
}

IOSurfaceRef ane_interop_copy_output(ANEHandle *handle, int index) {
    IOSurfaceRef s = ane_interop_get_output(handle, index);
    if (!s) return NULL;
    CFRetain(s);
    return s;
}

void ane_interop_free(ANEHandle *handle) {
    if (!handle) return;

    if (handle->realtimeLoaded && handle->client && handle->clientModel) {
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        id options = handle->evalOptions ? (__bridge id)handle->evalOptions : @{};
        NSError *rtErr = nil;
        if ([client respondsToSelector:@selector(unloadRealTimeModel:options:qos:error:)]) {
            ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                client, @selector(unloadRealTimeModel:options:qos:error:), modelObj, options, 21, &rtErr);
        }
        if ([client respondsToSelector:@selector(endRealTimeTask)]) {
            ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
        }
        handle->realtimeLoaded = false;
    }

    if (handle->model) {
        id mdl = (__bridge id)handle->model;
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
    }

    if (handle->ioInputs) {
        for (int i = 0; i < handle->nInputs; i++) {
            if (handle->ioInputs[i]) CFRelease(handle->ioInputs[i]);
        }
    }
    if (handle->ioOutputs) {
        for (int i = 0; i < handle->nOutputs; i++) {
            if (handle->ioOutputs[i]) CFRelease(handle->ioOutputs[i]);
        }
    }
    if (handle->tmpDir) {
        ane_interop_remove_tmpdir((__bridge id)handle->tmpDir);
    }

    if (handle->model) CFRelease(handle->model);
    if (handle->client) CFRelease(handle->client);
    if (handle->clientModel) CFRelease(handle->clientModel);
    if (handle->request) CFRelease(handle->request);
    if (handle->perfStats) CFRelease(handle->perfStats);
    if (handle->evalOptions) CFRelease(handle->evalOptions);
    if (handle->tmpDir) CFRelease(handle->tmpDir);

    free(handle->ioInputs);
    free(handle->ioOutputs);
    free(handle->inputBytes);
    free(handle->outputBytes);
    if (handle->liveHandleCounted) {
        __sync_fetch_and_sub(&g_live_handle_count, 1);
    }
    free(handle);
}

int ane_interop_compile_count(void) {
    return __sync_fetch_and_add(&g_compile_count, 0);
}

void ane_interop_set_compile_count(int value) {
    __sync_lock_test_and_set(&g_compile_count, value);
}

int ane_interop_last_compile_error(void) {
    return __sync_fetch_and_add(&g_last_compile_error, 0);
}

void ane_interop_set_force_eval_failure(bool value) {
    __sync_lock_test_and_set(&g_force_eval_failure, value ? 1 : 0);
}

int ane_interop_live_handle_count(void) {
    return __sync_fetch_and_add(&g_live_handle_count, 0);
}

uint64_t ane_interop_last_hw_execution_time_ns(ANEHandle *handle) {
    if (!handle) return 0;
    return handle->lastHwExecutionTimeNS;
}

bool ane_interop_has_perf_stats(ANEHandle *handle) {
    if (!handle) return false;
    return handle->perfStatsRequested && handle->perfStats != NULL;
}
