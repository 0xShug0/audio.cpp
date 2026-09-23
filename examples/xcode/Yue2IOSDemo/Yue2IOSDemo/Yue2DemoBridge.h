#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Yue2GenerationResult : NSObject

@property(nonatomic, copy) NSString *outputPath;
@property(nonatomic) double generationSeconds;
@property(nonatomic) double audioSeconds;
@property(nonatomic) double rtf;

@end

@interface Yue2DemoBridge : NSObject

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   backend:(NSString *)backend
                                    device:(int)device
                                   threads:(int)threads
                                     error:(NSError **)error;

- (nullable Yue2GenerationResult *)generateWithStyle:(NSString *)style
                                             lyrics:(NSString *)lyrics
                                                abc:(NSString *)abc
                                            cotMode:(NSString *)cotMode
                                         outputPath:(NSString *)outputPath
                                               seed:(uint64_t)seed
                                     inferenceSteps:(int64_t)inferenceSteps
                                  semanticMaxTokens:(int64_t)semanticMaxTokens
                                              error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
