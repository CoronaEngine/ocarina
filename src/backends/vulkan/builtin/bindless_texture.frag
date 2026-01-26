// Copyright 2020 Google LLC
#include "common.hlsl"
#include "push_constant.hlsl"

[[vk::binding(0, MATERIAL_SET)]]SamplerState sampler_albedo;
[[vk::binding(0, 2)]]Texture2D g_textures[];

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput
{
[[vk::location(0)]] float2 UV : TEXCOORD0;
[[vk::location(1)]] float4 Color : COLOR0;
};

float4 main(VSOutput input) : SV_TARGET
{
	float4 color = g_textures[pushConstants.albedoIndex].Sample(sampler_albedo, input.UV);

	return float4(color.rgb, 1.0);
}