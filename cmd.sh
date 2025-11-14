# TARGET=mc2hc3
# python s1model.py --name $TARGET --filter0 12 --filter1 8 --filter2 8 --filter3 6 --prune3 0.2
# python s2hlsmodelZ2.py --sambung --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc2hc4
# python s1model.py --name $TARGET --filter0 12 --filter1 8 --filter2 8 --filter3 6 --prune3 0.4
# python s2hlsmodelZ2.py --sambung --input=keras/mc2hc3/main_full.keras --output=$TARGET --minim

# TARGET=mc2hc5
# python s1model.py --name $TARGET --filter0 10 --filter1 8 --filter2 6 --filter3 6 --prune3 0.2
# python s2hlsmodelZ2.py --sambung --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc2hc6
# python s1model.py --name $TARGET --filter0 10 --filter1 8 --filter2 6 --filter3 6 --prune3 0.4
# python s2hlsmodelZ2.py --sambung --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc2hc7
# python s1model.py --name $TARGET --filter0 8 --filter1 6 --filter2 6 --filter3 6 --prune2 0.2 --prune3 0.4
# python s2hlsmodelZ2.py --sambung --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc2hc8
# python s1model.py --name $TARGET --filter0 8 --filter1 6 --filter2 6 --filter3 6 --prune2 0.5 --prune3 0.5
# python s2hlsmodelZ2.py --sambung --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc2hc9
# python s1model.py --name $TARGET --filter0 8 --filter1 6 --filter2 3 --filter3 6 --prune2 0.5 --prune3 0.5
# python s2hlsmodelZ2.py --sambung --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c1
# python s1modelc10.py --name $TARGET --filter0 32 --filter1 64 --filter2 128 --filter3 64

# TARGET=mc10c0b
# python s1modelc10.py --name $TARGET --filter0 48 --filter1 96 --filter2 192 --filter3 96

# TARGET=mc10c0c
# python s1modelc10.py --name $TARGET --filter0 48 --filter1 96 --filter2 192 --filter3 96

# TARGET=mc10c0d
# python s1modelc10.py --name $TARGET --filter0 48 --filter1 96 --filter2 192 --filter3 96

# TARGET=mc10c0e
# python s1modelc10.py --name $TARGET --filter0 48 --filter1 96 --filter2 192 --filter3 96

# TARGET=mc10c0f
# python s1modelc10.py --name $TARGET --filter0 32 --filter1 64 --filter2 128 --filter3 64

# TARGET=mc10c0g
# python s1modelc10.py --name $TARGET --filter0 48 --filter1 96 --filter2 192 --filter3 96
# python s2hlsmodelZ2.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c0h
# python s1modelc10.py --name $TARGET --filter0 48 --filter1 96 --filter2 192 --filter3 96
# python s2hlsmodelZ2.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c1
# python s1modelc10.py --name $TARGET --filter0 32 --filter1 64 --filter2 128 --filter3 64
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c1b
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/mc10c1/main_full.keras --output=$TARGET --minim

# TARGET=mc10c2
# python s1modelc10.py --name $TARGET --filter0 32 --filter1 32 --filter2 64 --filter3 32
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c3
# python s1modelc10.py --name $TARGET --filter0 16 --filter1 16 --filter2 32 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c4
# python s1modelc10.py --name $TARGET --filter0 8 --filter1 16 --filter2 16 --filter3 8
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c5
# python s1modelc10.py --name $TARGET --filter0 8 --filter1 20 --filter2 24 --filter3 8
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c6
# python s1modelc10.py --name $TARGET --filter0 8 --filter1 32 --filter2 64 --filter3 8
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c7
# python s1modelc10.py --name $TARGET --filter0 8 --filter1 28 --filter2 54 --filter3 7
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim

# TARGET=mc10c8
# python s1modelc10.py --name $TARGET --filter0 8 --filter1 20 --filter2 48 --filter3 7
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim
# python s3hls.py --name $TARGET

# TARGET=mc10c9
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 64 --filter3 32
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim
# python s3hls.py --name $TARGET

# TARGET=mc10c10
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 48 --filter3 24
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim
# python s3hls.py --name $TARGET

# TARGET=mc10c11
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 48 --filter3 24
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim
# python s3hls.py --name $TARGET

# TARGET=mc10c12
# python s1modelc10.py --name $TARGET --filter0 11 --filter1a 22 --filter1b 28 --filter2 42 --filter3 20
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim
# python s3hls.py --name $TARGET

# TARGET=mc10c13
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim
# python s3hls.py --name $TARGET

# TARGET=mc10c14
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-
# python s3hls.py --name $TARGET

# TARGET=mc10c15
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-
# python s3hls.py --name $TARGET

# TARGET=mc10c16
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c17
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c18
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c19
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c20
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c21
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c22
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c23
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c24
# # python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/mc10c23/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c25
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/{TARGET}/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c26
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 20 --filter1b 24 --filter2 40 --filter3 18
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/mc10c23/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c27
# python s1modelc10.py --name $TARGET --filter0 10 --filter1a 15 --filter1b 20 --filter2 32 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c28
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 64 --filter3 24
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c29
# python s1modelc10.py --name $TARGET --filter0 32 --filter1a 56 --filter1b 64 --filter2 128 --filter3 64
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c30
# python s1modelc10.py --name $TARGET --filter0 16 --filter1a 24 --filter1b 32 --filter1c 32 --filter2 64 --filter3 48
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c31
# python s1modelc10.py --name $TARGET --filter0 24 --filter1a 56 --filter1b 64 --filter2 128 --filter3 48 --prune2 0.3
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c32
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 48 --filter2 64 --filter3 48 --prune2 0.3
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c33
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 64 --filter3 32 --prune2 0.3
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c35
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 96 --filter3 32 --prune2 0.4
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis


# TARGET=mc10c36
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 64 --filter3 20
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c37
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 32 --filter2 56 --filter3 20
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c38
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 24 --filter2 56 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

#filter2 ke filter3 ternyata yang paling menahan slack

# TARGET=mc10c39
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 24 --filter2 56 --filter3 16 --prune2 0.4
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c40
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 24 --filter2 50 --filter3 14
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c41
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 24 --filter2 50 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c42
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 22 --filter1b 24 --filter2 50 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c43
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 22 --filter1b 22 --filter2 50 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c44
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 20 --filter1b 22 --filter2 50 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# TARGET=mc10c44
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 20 --filter1b 20 --filter2 50 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis


# lanjut dari mc10c42
# TARGET=mc10c45
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 22 --filter1b 24 --filter2 50 --filter3 16 \
#     --prune0 0.3 --prune1a 0.4 --prune1b 0.4 --prune2 0.5 --prune3 0.5
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET --novitis

# structured pruning - full filter prune

# Prune konservatif di layer awal (RECOMMENDED)
# TARGET=mc10c46
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 12 --prune0 0.0 \
#     --filter1a 22 --prune1a 0.2 \
#     --filter1b 24 --prune1b 0.3 \
#     --filter2 50 --prune2 0.5 \
#     --filter3 16 --prune3 0.2
# # Hasil: [12, 17, 16, 25, 12] filters - lebih seimbang
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# #  Start dengan filter lebih banyak untuk prune agresif
# TARGET=mc10c47
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 18 --prune0 0.3 \
#     --filter1a 36 --prune1a 0.4 \
#     --filter1b 40 --prune1b 0.4 \
#     --filter2 100 --prune2 0.5 \
#     --filter3 32 --prune3 0.5
# # Hasil: [12, 21, 24, 50, 16] filters - sama target tapi lebih smooth
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET


# TARGET=mc10c48 -- reuse factornya salah
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 18 --prune0 0.3 \
#     --filter1a 36 --prune1a 0.4 \
#     --filter1b 40 --prune1b 0.4 \
#     --filter2 100 --prune2 0.5 \
#     --filter3 32 --prune3 0.5
# # Hasil: [12, 21, 24, 50, 16] filters - sama target tapi lebih smooth
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
#     --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET

# TARGET=mc10c49
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 18 --prune0 0.3 \
#     --filter1a 36 --prune1a 0.4 \
#     --filter1b 40 --prune1b 0.4 \
#     --filter2 100 --prune2 0.6 \
#     --filter3 32 --prune3 0.5
# # Hasil: [12, 21, 24, 50, 16] filters - sama target tapi lebih smooth
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
#     --output=$TARGET --minim --skip-profiling \
#     --reuse0 432 \
#     --reuse1a 864 \
#     --reuse1b 960 \
#     --reuse2 2400 \
#     --reuse3 1536
# python s3hls.py --name $TARGET

# TARGET=mc10c50
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 18 --prune0 0.3 \
#     --filter1a 36 --prune1a 0.5 \
#     --filter1b 40 --prune1b 0.5 \
#     --filter2 100 --prune2 0.7 \
#     --filter3 32 --prune3 0.6
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
#     --output=$TARGET --minim --skip-profiling \
#     --reuse0 432 \
#     --reuse1a 432 \
#     --reuse1b 432 \
#     --reuse2 432 \
#     --reuse3 432
# python s3hls.py --name $TARGET

# TARGET=mc10c50b
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/mc10c50/main_full.keras \
#     --output=$TARGET --minim --skip-profiling \
#     --reuse0 232 \
#     --reuse1a 232 \
#     --reuse1b 232 \
#     --reuse2 232 \
#     --reuse3 232
# python s3hls.py --name $TARGET

# TARGET=mc10c51
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 18 --prune0 0.35 \
#     --filter1a 36 --prune1a 0.6 \
#     --filter1b 40 --prune1b 0.6 \
#     --filter2 100 --prune2 0.75 \
#     --filter3 32 --prune3 0.65
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
#     --output=$TARGET --minim --skip-profiling \
#     --reuse0 232 \
#     --reuse1a 232 \
#     --reuse1b 232 \
#     --reuse2 232 \
#     --reuse3 232
# python s3hls.py --name $TARGET


# TARGET=mc10c52
# python s1modelc10_filterpruned.py --name $TARGET \
#     --filter0 18 --prune0 0.3 \
#     --filter1a 36 --prune1a 0.54 \
#     --filter1b 40 --prune1b 0.54 \
#     --filter2 100 --prune2 0.75 \
#     --filter3 32 --prune3 0.65
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
#     --output=$TARGET --minim --skip-profiling \
#     --reuse0 432 \
#     --reuse1a 432 \
#     --reuse1b 432 \
#     --reuse2 432 \
#     --reuse3 432
# python s3hls.py --name $TARGET
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
#     --output=$(TARGET)b --minim --skip-profiling \
#     --reuse0 232 \
#     --reuse1a 232 \
#     --reuse1b 232 \
#     --reuse2 232 \
#     --reuse3 232
# python s3hls.py --name $(TARGET)b





# TARGET=mc10c38
# python s1modelc10.py --name $TARGET --filter0 12 --filter1a 24 --filter1b 24 \
#    --filter2 56 --filter3 16
# python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras --output=$TARGET --minim --skip-profiling
# python s3hls.py --name $TARGET



TARGET=mc10c53
python s1modelc10_filterpruned.py --name $TARGET \
    --filter0 18 --prune0 0.33 \
    --filter1a 36 --prune1a 0.33 \
    --filter1b 40 --prune1b 0.4 \
    --filter2 100 --prune2 0.44 \
    --filter3 32 --prune3 0.5
python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
    --output=$TARGET --minim --skip-profiling \
    --reuse0 232 \
    --reuse1a 232 \
    --reuse1b 232 \
    --reuse2 232 \
    --reuse3 232
python s3hls.py --name $TARGET

TARGET=mc10c54
python s1modelc10_filterpruned.py --name $TARGET \
    --filter0 18 --prune0 0.33 \
    --filter1a 36 --prune1a 0.5 \
    --filter1b 40 --prune1b 0.5 \
    --filter2 100 --prune2 0.5 \
    --filter3 32 --prune3 0.5
python s2hlsmodelZ2c10.py --sambung --c10 --input=keras/$TARGET/main_full.keras \
    --output=$TARGET --minim --skip-profiling \
    --reuse0 232 \
    --reuse1a 232 \
    --reuse1b 232 \
    --reuse2 232 \
    --reuse3 232
python s3hls.py --name $TARGET
