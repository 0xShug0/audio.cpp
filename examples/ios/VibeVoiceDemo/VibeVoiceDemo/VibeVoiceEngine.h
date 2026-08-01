#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface VibeVoiceGenerationResult : NSObject

@property(nonatomic, strong, readonly) NSURL *wavURL;
@property(nonatomic, assign, readonly) double generationSeconds;
@property(nonatomic, assign, readonly) double audioSeconds;
@property(nonatomic, assign, readonly) double rtf;

- (instancetype)initWithWavURL:(NSURL *)wavURL
             generationSeconds:(double)generationSeconds
                  audioSeconds:(double)audioSeconds
                           rtf:(double)rtf NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

typedef void (^VibeVoiceGenerationCompletion)(
    VibeVoiceGenerationResult *_Nullable result,
    NSError *_Nullable error);

@interface VibeVoiceEngine : NSObject

+ (instancetype)sharedEngine NS_SWIFT_NAME(shared());

- (void)generateWithText:(NSString *)text
                modelURL:(NSURL *)modelURL
               voiceURLs:(NSArray<NSURL *> *)voiceURLs
          inferenceSteps:(NSInteger)inferenceSteps
              completion:(VibeVoiceGenerationCompletion)completion;

@end

NS_ASSUME_NONNULL_END
