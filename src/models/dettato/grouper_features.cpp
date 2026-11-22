#include "grouper.h"
#include "frontend.h"
#include "windows.h"
#include <algorithm>
#include <cmath>
#include <tuple>

namespace engine::models::dettato::detail {
GroupingFeatures collect_grouping_features(const WindowTranscription & transcription,const std::vector<Note> & notes,
    const NoteThresholds & thresholds) {
    const auto T=transcription.frames;
    if(T<=0 || transcription.grouping_hidden->size()!=(transcription.compact_grouping ? transcription.grouping_rows.size()*368 : static_cast<size_t>(T*88*368)) ||
        transcription.grouping_stem->size()!=transcription.grouping_hidden->size())
        throw std::runtime_error("Dettato grouping requires aligned hidden/stem sub-averages");
    const auto row_offset = [&](int64_t t, int64_t lane) {
        const auto row = t * 88 + lane;
        if (!transcription.compact_grouping) return static_cast<size_t>(row * 368);
        const auto found = std::lower_bound(transcription.grouping_rows.begin(), transcription.grouping_rows.end(), row);
        if (found == transcription.grouping_rows.end() || *found != row)
            throw std::runtime_error("Dettato grouping readback omitted a required feature row");
        return static_cast<size_t>(found - transcription.grouping_rows.begin()) * 368;
    };
    const auto own=decode_notes(transcription.grouping_maps,T,.02,thresholds);
    auto sorted=notes;
    std::stable_sort(sorted.begin(),sorted.end(),[](const auto & a,const auto & b) {return std::tie(a.onset,a.pitch)<std::tie(b.onset,b.pitch);});
    const size_t count=sorted.size();
    GroupingFeatures out;out.notes.resize(count*12);out.taps.resize(count*1104);
    std::vector<int64_t> a(count),b(count),lanes(count);
    std::array<std::vector<size_t>,88> selected;
    const auto clip=[&](int64_t frame) {return std::clamp<int64_t>(frame,0,T-1);};
    for(size_t i=0;i<count;++i) {
        const auto & note=sorted[i];
        const double off=std::max(note.offset,note.onset+.02);
        double closest=INFINITY;int slot=0;
        for(const auto & candidate:own) if(candidate.pitch==note.pitch) {
            const double distance=std::abs(candidate.onset-note.onset);
            if(distance<closest) {closest=distance;slot=candidate.voice-1;}
        }
        if(closest>.1) slot=0;
        a[i]=clip(static_cast<int64_t>(std::nearbyint(note.onset/.02)));
        b[i]=std::min<int64_t>(std::max<int64_t>(static_cast<int64_t>(std::ceil(off/.02)),a[i]+1),T);
        lanes[i]=std::clamp<int64_t>(note.pitch-21,0,87);selected[static_cast<size_t>(lanes[i])].push_back(i);
        auto * meta=out.notes.data()+i*12;
        meta[0]=static_cast<float>(note.onset);meta[1]=static_cast<float>(off);meta[2]=static_cast<float>(note.pitch);meta[3]=static_cast<float>(slot);
        const auto get=[&](int head,int64_t t) {return transcription.grouping_maps[head][static_cast<size_t>(clip(t)*88+lanes[i])];};
        const auto maximum=[&](int head,int64_t t) {return std::max({get(head,t-1),get(head,t),get(head,t+1)});};
        meta[4]=maximum(0,a[i]);meta[5]=maximum(3,a[i]);meta[6]=get(1,a[i]);meta[7]=get(4,a[i]);
        meta[10]=maximum(2,b[i]-1);meta[11]=maximum(5,b[i]-1);
        std::array<float,1104> values{};
        const auto limit=std::min<int64_t>(b[i],a[i]+50),stem_limit=std::min<int64_t>(b[i],a[i]+10);
        ggml_fp16_to_fp32_row(transcription.grouping_hidden->data()+row_offset(a[i],lanes[i]),values.data(),368);
        std::array<float,368> row;
        for(int64_t t=a[i];t<limit;++t) {
            ggml_fp16_to_fp32_row(transcription.grouping_hidden->data()+row_offset(t,lanes[i]),row.data(),368);
            for(size_t channel=0;channel<368;++channel) values[368+channel]+=row[channel];
        }
        for(int64_t t=a[i];t<stem_limit;++t) {
            ggml_fp16_to_fp32_row(transcription.grouping_stem->data()+row_offset(t,lanes[i]),row.data(),368);
            for(size_t channel=0;channel<368;++channel) values[736+channel]+=row[channel];
        }
        for(size_t channel=0;channel<368;++channel) {
            values[368+channel]/=static_cast<float>(limit-a[i]);
            values[736+channel]/=static_cast<float>(stem_limit-a[i]);
        }
        for(size_t tap=0;tap<3;++tap) {
            float maximum=0;
            for(size_t c=0;c<368;++c) maximum=std::max(maximum,std::abs(values[tap*368+c]));
            const float scale=std::max(maximum,1e-6f)/127;
            const float stored_scale=ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
            for(size_t c=0;c<368;++c) {
                const float value=std::clamp(std::nearbyint(values[tap*368+c]/scale),-127.0f,127.0f);
                out.taps[(i*3+tap)*368+c]=ggml_fp32_to_fp16(value*stored_scale);
            }
        }
    }
    std::vector<float> prefix(static_cast<size_t>((T+1)*2));
    for(size_t lane=0;lane<selected.size();++lane) {
        if(selected[lane].empty()) continue;
        for(size_t voice=0;voice<2;++voice) {
            double sum=0;prefix[voice*(T+1)]=0;
            for(int64_t t=0;t<T;++t) {sum+=transcription.grouping_maps[1+voice*3][static_cast<size_t>(t*88+lane)];prefix[voice*(T+1)+t+1]=static_cast<float>(sum);}
            for(const auto i:selected[lane]) out.notes[i*12+8+voice]=(prefix[voice*(T+1)+b[i]]-prefix[voice*(T+1)+a[i]])/static_cast<float>(b[i]-a[i]);
        }
    }
    return out;
}

GrouperInputs make_grouper_inputs(const HarmonicFeatures & features,const std::vector<float> & notes,
    const std::vector<float> & taps,const std::vector<float> & harmonic_prior,const std::vector<float> & lane_reads) {
    const int64_t T=features.frames,E=features.extended_bins;
    if(features.batch!=1 || T<=0 || E<=0 || notes.size()%12 || harmonic_prior.size()!=10 ||
        features.values.size()!=static_cast<size_t>(9*T*288) || features.power.size()!=static_cast<size_t>(E*T))
        throw std::runtime_error("Dettato grouping readers require aligned single-window features and notes");
    const size_t count=notes.size()/12;
    if(taps.size()!=count*1104 || lane_reads.size()!=count*256) throw std::runtime_error("Dettato grouping tap/read rows differ from notes");
    GrouperInputs out;
    out.taps=taps;out.lanes=lane_reads;out.pitch.resize(count);out.scalars.resize(count*11);
    out.harmonics.resize(count*150);out.profile.resize(count*36);out.cosine.resize(count*4*22);out.sine.resize(out.cosine.size());
    std::vector<int64_t> a(count),b(count),lane(count);
    std::vector<int64_t> bins(count*10);
    std::array<float,10> prior;
    std::array<int64_t,10> shifts;
    for(size_t h=0;h<10;++h) {
        const float value=harmonic_prior[h];
        prior[h]=std::max(value,0.0f)+std::log1p(std::exp(-std::abs(value)));
        shifts[h]=static_cast<int64_t>(std::nearbyint(36*std::log2(static_cast<double>(h+1))));
    }
    std::vector<float> occupancy(static_cast<size_t>(E*(T+1)),0);
    for(size_t i=0;i<count;++i) {
        const auto * note=notes.data()+i*12;
        out.pitch[i]=std::clamp<int32_t>(static_cast<int32_t>(std::nearbyint(note[2])),0,127);
        lane[i]=std::clamp<int64_t>(out.pitch[i]-21,0,87);
        a[i]=std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint((note[0]+.5f)*100)),0,T-1);
        b[i]=std::max<int64_t>(std::min<int64_t>(static_cast<int64_t>(std::nearbyint((note[1]+.5f)*100)),T),a[i]+1);
        out.scalars[i*11]=std::log(std::max(note[1]-note[0],.01f));
        std::copy_n(note+3,9,out.scalars.begin()+i*11+1);out.scalars[i*11+10]=note[2]/64-1;
        for(size_t h=0;h<10;++h) {
            bins[i*10+h]=3*lane[i]+36+shifts[h];
            if(bins[i*10+h]<E) occupancy[static_cast<size_t>(bins[i*10+h]*(T+1)+a[i])]+=prior[h];
        }
        for(int64_t head=0;head<4;++head) for(int64_t d=0;d<22;++d) {
            const float frequency=1/std::pow(10000.0f,static_cast<float>(d)/22);
            const float angle=(note[0]*10)*frequency;
            const auto index=static_cast<size_t>((head*count+i)*22+d);
            out.cosine[index]=std::cos(angle);out.sine[index]=std::sin(angle);
        }
    }
    // Match the two scatter operations: all starts, then all ends.
    for(size_t i=0;i<count;++i) for(size_t h=0;h<10;++h)
        if(bins[i*10+h]<E) occupancy[static_cast<size_t>(bins[i*10+h]*(T+1)+b[i])]-=prior[h];
    for(int64_t bin=0;bin<E;++bin) {
        double sum=0;
        for(int64_t t=0;t<T;++t) {sum+=occupancy[static_cast<size_t>(bin*(T+1)+t)];occupancy[static_cast<size_t>(bin*(T+1)+t)]=static_cast<float>(sum);}
    }
    const std::array<int,10> attacks{-2,0,2,4,8,12,16,24,32,48};
    const std::array<float,4> sustain{.25f,.5f,.75f,.95f};
    for(size_t i=0;i<count;++i) {
        std::array<int64_t,14> time;
        for(size_t k=0;k<attacks.size();++k) time[k]=std::clamp<int64_t>(a[i]+attacks[k],0,T-1);
        for(size_t k=0;k<sustain.size();++k) time[10+k]=std::clamp<int64_t>(a[i]+static_cast<int64_t>((b[i]-a[i])*sustain[k]),0,T-1);
        float maximum=-INFINITY;
        for(size_t h=0;h<10;++h) {
            const bool valid=bins[i*10+h]<E;
            const auto bin=std::clamp<int64_t>(bins[i*10+h],0,E-1);
            out.harmonics[i*150+140+h]=valid ? 1.0f : 0.0f;
            for(size_t k=0;k<time.size();++k) {
                const float active=time[k]>=a[i] && time[k]<b[i] ? 1.0f : 0.0f;
                const float total=occupancy[static_cast<size_t>(bin*(T+1)+time[k])]+prior[h]*(1-active);
                const float share=std::min(prior[h]/std::max(total,1e-6f),1.0f);
                const float value=std::log(features.power[static_cast<size_t>(bin*T+time[k])]*share+1e-7f)*(valid ? 1.0f : 0.0f);
                out.harmonics[i*150+h*14+k]=value;maximum=std::max(maximum,value);
            }
        }
        for(size_t j=0;j<140;++j) out.harmonics[i*150+j]=std::max(out.harmonics[i*150+j]-maximum,-20.0f)/10;
    }
    std::array<std::vector<std::pair<size_t,size_t>>,288> reads;
    for(size_t i=0;i<count;++i) for(size_t offset=0;offset<2;++offset)
        reads[static_cast<size_t>(std::min<int64_t>(3*lane[i]+offset,287))].push_back({i,offset});
    std::vector<float> sum(static_cast<size_t>(T+1)),square(sum.size());
    for(size_t channel=0;channel<9;++channel) for(size_t bin=0;bin<reads.size();++bin) {
        if(reads[bin].empty()) continue;
        double value_sum=0,square_sum=0;sum[0]=square[0]=0;
        for(int64_t t=0;t<T;++t) {
            const float value=features.values[static_cast<size_t>((channel*T+t)*288+bin)];
            value_sum+=value;square_sum+=static_cast<float>(value*value);
            sum[static_cast<size_t>(t+1)]=static_cast<float>(value_sum);square[static_cast<size_t>(t+1)]=static_cast<float>(square_sum);
        }
        for(const auto & read:reads[bin]) {
            const auto i=read.first,offset=read.second;
            const float duration=static_cast<float>(b[i]-a[i]);
            const float mean=(sum[static_cast<size_t>(b[i])]-sum[static_cast<size_t>(a[i])])/duration;
            const float second=(square[static_cast<size_t>(b[i])]-square[static_cast<size_t>(a[i])])/duration;
            out.profile[i*36+offset*18+channel]=mean;
            out.profile[i*36+offset*18+9+channel]=std::sqrt(std::max(second-mean*mean,0.0f)+1e-4f);
        }
    }
    return out;
}
} // namespace engine::models::dettato::detail
